# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import time 
from typing import List, Union
from types import MethodType
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
import traceback
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

from verl.workers.rollout.vllm_rollout_with_executor.parser import extract_program, extract_jupyter_like_program
from verl.workers.rollout.vllm_rollout_with_executor.python_executor import PythonExecutor


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRolloutWithExecutor(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.actor_module = actor_module
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        self.tensor_parallel_size = tensor_parallel_size
        
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)
        
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.val_response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(actor_module,
                                    tokenizer=tokenizer,
                                    model_hf_config=model_hf_config,
                                    tensor_parallel_size=tensor_parallel_size,
                                    dtype=config.dtype,
                                    enforce_eager=config.enforce_eager,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    skip_tokenizer_init=False,
                                    max_model_len=max(config.prompt_length + config.response_length, config.prompt_length + config.val_response_length),
                                    max_num_batched_tokens=max_num_batched_tokens,
                                    enable_chunked_prefill=config.enable_chunked_prefill,
                                    load_format=config.load_format)

        # override the default post_process_outputs by verl; use the direct outputs from vllm
        def _post_process_outputs(self, request_outputs):
            return request_outputs

        self.inference_engine._post_process_outputs = MethodType(_post_process_outputs, self.inference_engine)
        
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False
        
        # force n to be 1 
        kwargs['n'] = 1
        print(f"rollout with exector will set n = {self.config.n} to 1 in sampling params. repeated sampling is specifically handled.")

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        self.executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=5)
        # self.max_num_func_calls = config.get('max_num_func_calls', 7)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, max_retries: int = 1e9, **kwargs) -> DataProto:
        """Generate sequences using vLLM engine with retry logic for failures.

        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids and meta_info.
            max_retries (int, optional): Maximum number of retries on failure. Defaults to 1e9.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Generated response token ids
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence

        Raises:
            RuntimeError: If generation fails after max_retries attempts.
        """
        max_retries = int(max_retries)
        for attempt in range(max_retries):
            try:
                # Rebuild vLLM cache engine if configured
                if self.config.free_cache_engine:
                    self.inference_engine.init_cache_engine()

                # Extract input tensors from prompt batch
                idx = prompts.batch['input_ids']
                attention_mask = prompts.batch['attention_mask']
                position_ids = prompts.batch['position_ids']
                eos_token_id = prompts.meta_info['eos_token_id']
                batch_size = idx.size(0)

                # Pre-process input token ids
                idx_list = [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ]

                # Configure sampling parameters
                do_sample = prompts.meta_info.get('do_sample', True)
                if not do_sample:
                    kwargs = {
                        'best_of': 1,
                        'top_p': 1.0,
                        'top_k': -1,
                        'min_p': 0.0,
                        'temperature': 0,
                        'n': 1
                    }
                is_in_val = False
                if prompts.meta_info.get('val_temperature', None):
                    kwargs['temperature'] = prompts.meta_info['val_temperature']
                    kwargs['max_tokens'] = prompts.meta_info['val_response_length']
                    is_in_val = True
                # Generate sequences
                with self.update_sampling_params(**kwargs):
                    if prompts.meta_info.get('customized_do_sample', False):
                        do_sample_list = prompts.non_tensor_batch['do_sample']
                        sampling_params_list = []
                        assert len(do_sample_list) == batch_size, "{} != {}".format(len(do_sample_list), batch_size)
                        for i in range(len(do_sample_list)):
                            sampling_params_list.append(self.sampling_params.clone())
                            sampling_params_list[-1].n = 1
                            if not do_sample_list[i]:
                                sampling_params_list[-1].best_of = 1
                                sampling_params_list[-1].top_p = 1.0
                                sampling_params_list[-1].top_k = -1
                                sampling_params_list[-1].min_p = 0.0
                                sampling_params_list[-1].temperature = 0
                        
                        assert len(sampling_params_list) == len(idx_list), "{} != {}".format(len(sampling_params_list), len(idx_list))

                        output = self._generate_with_executor(
                            prompt_token_ids=idx_list,
                            sampling_params=sampling_params_list,
                            verbose=False, 
                            is_in_val=is_in_val
                        )
                        
                        do_sample = False # for later use
                    else:
                        output = self._generate_with_executor(
                            prompt_token_ids=idx_list,
                            sampling_params=self.sampling_params,
                            verbose=False,
                            is_in_val=is_in_val,
                            n=1 if not do_sample else None   # for n to be 1 if not do_sample
                        )

                # Process outputs
                response = output[0].to(idx.device)
                log_probs = output[1].to(idx.device)

                # Pad sequences if needed
                if is_in_val:
                    if response.shape[1] < self.config.val_response_length:
                        response = pad_sequence_to_length(
                            response, self.config.val_response_length, self.pad_token_id)
                        log_probs = pad_sequence_to_length(
                            log_probs, self.config.val_response_length, self.pad_token_id)
                else:
                    if response.shape[1] < self.config.response_length:
                        response = pad_sequence_to_length(
                            response, self.config.response_length, self.pad_token_id)
                        log_probs = pad_sequence_to_length(
                            log_probs, self.config.response_length, self.pad_token_id)

                # Handle multiple samples per prompt
                if self.config.n > 1 and do_sample:
                    idx = idx.repeat_interleave(self.config.n, dim=0)
                    attention_mask = attention_mask.repeat_interleave(
                        self.config.n, dim=0)
                    position_ids = position_ids.repeat_interleave(
                        self.config.n, dim=0)
                    batch_size = batch_size * self.config.n

                # Concatenate prompt and response
                seq = torch.cat([idx, response], dim=-1)

                # Create position IDs and attention mask for full sequence
                response_length = response.size(1)
                delta_position_id = torch.arange(
                    1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(
                    batch_size, 1)

                response_position_ids = position_ids[:, -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids],
                                       dim=-1)
                response_attention_mask = get_eos_mask(
                    response_id=response,
                    eos_token=eos_token_id,
                    dtype=attention_mask.dtype)
                attention_mask = torch.cat(
                    (attention_mask, response_attention_mask), dim=-1)

                # Construct output batch
                batch = TensorDict(
                    {
                        'prompts': idx,
                        'responses': response,
                        'input_ids': seq,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids
                    },
                    batch_size=batch_size)

                # Free cache if configured
                if self.config.free_cache_engine:
                    self.inference_engine.free_cache_engine()

                return DataProto(batch=batch)

            except Exception as e:
                traceback.print_exc()
                print(f"error.trackback: {e.traceback}")
                print("Restarting vLLM due to error: ", e)
                print("Retrying...")

                # Clean up and restart engine
                torch.cuda.empty_cache()
                if hasattr(self.inference_engine, 'free_cache_engine'):
                    self.inference_engine.free_cache_engine()
                del self.inference_engine

                # Reinitialize engine with same parameters
                self.inference_engine = LLM(
                    self.actor_module,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.model_hf_config,
                    tensor_parallel_size=self.tensor_parallel_size,
                    dtype=self.config.dtype,
                    enforce_eager=self.config.enforce_eager,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    skip_tokenizer_init=False,
                    max_model_len=max(self.config.prompt_length + self.config.response_length, self.config.prompt_length + self.config.val_response_length),
                    load_format=self.config.load_format)
                print("vLLM is ready to roll!")

                if attempt < max_retries - 1:
                    continue

        raise RuntimeError(
            f"Failed to generate sequences after {max_retries} attempts")

    def _generate_with_executor(self, prompt_token_ids: List[List[int]], sampling_params: Union[SamplingParams, List[SamplingParams]], n: int = None, verbose=False, is_in_val=False) -> DataProto:
        if not isinstance(sampling_params, list):
            new_sampling_param = sampling_params.clone()
            new_sampling_param.n = n if n is not None else self.config.n
            sampling_params = [new_sampling_param] * len(prompt_token_ids)
            print("-" * 20, "Updated sampling params", "-" * 20)
            print(sampling_params[0]) 
        
        assert len(prompt_token_ids) == len(sampling_params), "{} != {}".format(len(prompt_token_ids), len(sampling_params))

        queries = [self.tokenizer.decode(prompt_token_ids[i]) for i in range(len(prompt_token_ids))]
        if verbose and torch.distributed.get_rank() == 0:
            print(f"is_in_val: {is_in_val}")
            print(f"queries: {[q[:64] for q in queries]}; len(queries): {len(queries)}")

        stop_word = "`\n"
        stop_token_ids = self.tokenizer.encode(stop_word, add_special_tokens=False)
        if verbose and torch.distributed.get_rank() == 0:
            print(f"stop_word: {stop_word}; stop_token_ids: {stop_token_ids}")
        assert len(stop_token_ids) == 1
        # repeat the queries n times
        queries_with_sampling_params = []
        for (query, sampling_param) in zip(queries, sampling_params):
            for _ in range(sampling_param.n):
                new_sampling_param = sampling_param.clone()
                new_sampling_param.n = 1
                new_sampling_param.stop_token_ids = stop_token_ids
                queries_with_sampling_params.append((query, new_sampling_param))
        if verbose and torch.distributed.get_rank() == 0:
            print(f"queries_with_sampling_params: {[q[0][:64] for q in queries_with_sampling_params]}; len(queries_with_sampling_params): {len(queries_with_sampling_params)}")

        if verbose and torch.distributed.get_rank() == 0:
            print(f"There are {len(queries)} queries with n = {self.config.n}")

        remain_sequences = [(i, query, sampling_param) for i, (query, sampling_param) in enumerate(queries_with_sampling_params)]
        end_sequences = []

        start_time = time.time()
        if is_in_val:
            max_num_func_calls = self.config.get('val_max_num_func_calls', 7)
        else:
            max_num_func_calls = self.config.get('max_num_func_calls', 7)
        if verbose and torch.distributed.get_rank() == 0:
            print(f"max_num_func_calls: {max_num_func_calls}")
        for epoch in range(max_num_func_calls):
            current_sequences = remain_sequences
            if len(current_sequences) == 0:
                break
            if verbose and torch.distributed.get_rank() == 0:
                print("=" * 50, "Epoch", epoch)

            # get all outputs
            prompts = [item[1] for item in current_sequences]
            # sampling_params = [item[2] for item in current_sequences]
            sampling_params = []
            for item in current_sequences:
                new_sampling_param = item[2].clone()
                if epoch == (max_num_func_calls - 1):
                    new_sampling_param.stop_token_ids = []
                sampling_params.append(new_sampling_param)

            idx_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]

            request_outputs = self.inference_engine.generate(
                prompt_token_ids=idx_list,
                sampling_params=sampling_params,
                use_tqdm=False
            )

            ##############################################################
            # generation with strings may lead to the problem of handling eos and stop tokens 
            ##############################################################
            # request_outputs = self.inference_engine.generate(
            #     prompts=prompts, 
            #     sampling_params=sampling_params,
            #     use_tqdm=False
            # )

            outputs = []
            output_tokens = []
            for output in request_outputs:
                for out in output.outputs:
                    completion_txt = out.text
                    if not out.text:
                        completion_txt = self.tokenizer.decode(out.token_ids)  # in this case, out.token_ids already include the stop token
                    else:
                        completion_txt = out.text
                        if out.finish_reason == "stop" and out.stop_reason in stop_token_ids:
                            completion_txt += self.tokenizer.decode(out.stop_reason)
                    if completion_txt is None:
                        outputs.append("")
                        output_tokens.append([])
                        if verbose and torch.distributed.get_rank() == 0:
                            print(f"warning of vllm_rollout_with_executor: get completion_txt is None!")
                    else:
                        outputs.append(completion_txt)
                        output_tokens.append(out.token_ids)
            assert len(outputs) == len(current_sequences), "{} != {}".format(len(outputs), len(current_sequences))

            # process all outputs
            remain_sequences = []
            remain_codes = []
            for (i, prompt, sampling_param), output in zip(current_sequences, outputs):
                prompt += output
                if output.endswith(stop_word):
                    if ("```python" in output):
                        program = extract_jupyter_like_program(prompt)
                        remain_sequences.append((i, prompt, sampling_param))
                        remain_codes.append((i, program))
                    else:
                        remain_sequences.append((i, prompt, sampling_param))
                        remain_codes.append((i, ""))
                else:
                    end_sequences.append((i, prompt))

            # execute the remain prompts
            remain_indices = []
            remain_results = self.executor.batch_apply([x[1] for x in remain_codes])
            for k in range(len(remain_codes)):
                i, program = remain_codes[k]
                res, report = remain_results[k]
                if res == "empty code":
                    exec_result = ""
                else:
                    exec_result = res if res else report
                    exec_result = str(exec_result)
                    # run out of python exe
                    if epoch == max_num_func_calls - 2:
                        exec_result += f"\n\n[SYSTEM]\nYou have exceeded the allowed number of code executions. You can no longer write or run code. Please continue solving the problem using your reasoning and analytical skills."
                    exec_result = f"```output\n{exec_result}\n```\n"
                if verbose and k == 0 and torch.distributed.get_rank() == 0:
                    print("*" * 20, "prompt", "*" * 20)
                    print(remain_sequences[k][-2])
                    print("*" * 20, "program", "*" * 20)
                    print(program)
                    print("*" * 20, "exec_result", "*" * 20)
                    print(exec_result)

                (j, prompt, sampling_param) = remain_sequences[k]
                prompt_without_exec_result = prompt[:]
                assert i == j, "i and j must be the same: {} != {}".format(i, j)

                prompt += exec_result

                # # not end
                # if epoch == max_num_func_calls - 1:
                #     prompt += f"```output\n[SYSTEM] Maximum allowed function calls reached. Python runtime has been terminated. Further code execution is permanently disabled.\n```\n"

                # check whether there are enough tokens for the remaining prompt
                query = queries_with_sampling_params[i][0]   # pure question
                query_length = len(self.tokenizer.encode(query, add_special_tokens=False))
                if is_in_val:
                    max_tokens = self.config.val_response_length + query_length - len(self.tokenizer.encode(prompt, add_special_tokens=False))
                else:
                    max_tokens = self.config.response_length + query_length - len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if max_tokens <= 0:
                    if verbose and k == 0 and torch.distributed.get_rank() == 0:
                        print("*" * 20, "skip this executation result due to length limit", "*" * 20)
                    remain_sequences[k] = (i, prompt_without_exec_result, sampling_param)
                    end_sequences.append((i, prompt_without_exec_result))
                else:
                    sampling_param.max_tokens = max_tokens
                    remain_sequences[k] = (i, prompt, sampling_param)
                    remain_indices.append(k)
        
            remain_sequences = [remain_sequences[i] for i in remain_indices]
    
        # unsolved samples
        if verbose and torch.distributed.get_rank() == 0:
            print("Unsolved samples:", len(remain_sequences))
        end_sequences.extend([x[:2] for x in remain_sequences])  # exclude the sampling_param

        # sort by idx
        end_sequences = sorted(end_sequences, key=lambda x: x[0])
        time_use = time.time() - start_time
        if verbose and torch.distributed.get_rank() == 0:
            print(f"Time use to generate {len(end_sequences)} sequences: {time_use:.2f}s")

        # put results back to examples
        completions = []
        for ((query, sampling_param), (i, sequence)) in zip(queries_with_sampling_params, end_sequences):
            initial_prompt = query
            initial_prompt_len = len(initial_prompt)
            completion = sequence[initial_prompt_len:]
            completions.append(completion)
        
        completion_token_ids = [torch.tensor(self.tokenizer.encode(completion, add_special_tokens=False)) for completion in completions]
        assert len(completion_token_ids) == len(queries_with_sampling_params), "{} != {}".format(len(completion_token_ids), len(queries))
        for i in range(len(completion_token_ids)):
            assert len(completion_token_ids[i]) > 0
            if is_in_val:
                if len(completion_token_ids[i]) > self.config.val_response_length:
                    print(f"warning: len(completion_token_ids[i]) = {len(completion_token_ids[i])} > self.config.val_response_length = {self.config.val_response_length}")
                    completion_token_ids[i] = completion_token_ids[i][:self.config.val_response_length]
            else:
                if len(completion_token_ids[i]) > self.config.response_length:
                    print(f"warning: len(completion_token_ids[i]) = {len(completion_token_ids[i])} > self.config.response_length = {self.config.response_length}")
                    completion_token_ids[i] = completion_token_ids[i][:self.config.response_length]

        logprobs = [torch.zeros(len(completion_token_ids[i])) for i in range(len(completion_token_ids))]

        # post-process to match the output format of verl
        pad_token_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id)

        completion_token_ids = pad_sequence(completion_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=0.0)
        
        return (completion_token_ids, logprobs)