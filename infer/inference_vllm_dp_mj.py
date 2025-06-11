"""
This script supports vllm batch inference with cot/pal/tora prompt.
Also supports inference of fine-tuned models like WizardMath/ToRA.
Uses original batch processing with traditional majority voting.
Code based on: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""
import random
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['RAY_num_server_call_thread'] = '1'
import json
import argparse
import time
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from transformers import AutoTokenizer
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from multiprocessing import Process
import timeout_decorator

from infer.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from infer.parser import *
from infer.python_executor import PythonExecutor
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
from math_verify import verify


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="gsm8k.jsonl", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--engine", default="vllm", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--n_sampling", default=8, type=int)  # Number of samples for majority voting
    parser.add_argument("--stop_tokens_mode", default="normal_code_block_end", type=str)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--data_parallel_size", default=1, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_model_len", default=130000, type=int)
    parser.add_argument("--max_tokens_per_call", default=32768, type=int)
    parser.add_argument("--max_func_call", default=10, type=int)
    parser.add_argument("--func_call_mode", default="jupyter", type=str)
    parser.add_argument("--func_call_timeout", default=30, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--enable_cache", action="store_true")
    parser.add_argument("--extract_majority_answer", action="store_true", help="Extract and include majority answer in results")
    parser.add_argument("--verbose_func_call_output", action="store_true", help="Extract and include majority answer in results")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    assert args.engine == "vllm"
    if args.stop_tokens_mode == "normal_code_block_end":
        args.stop = ["`\n"]
    elif args.stop_tokens_mode == "old_normal_code_block_end":
        args.stop = ["```\n", "``` \n", "```  \n", "```   \n", "```    \n", "```     \n", "```      \n", "```       \n", "```        \n"]
    else:
        args.stop = ["/python"]
    print(f"args detail:")
    print(args)
    time.sleep(5)
    return args


def prepare_data(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cache_prompts = set()
    if args.enable_cache:
        if os.path.exists(args.output_dir):
            output_dir_ents = os.listdir(args.output_dir)
            for de in output_dir_ents:
                if de.startswith("dp_rank"):
                    output_file = os.path.join(args.output_dir, de)
                    with open(output_file) as fd:
                        for line in fd:
                            example = json.loads(line)
                            cache_prompts.add(example["prompt"])

    examples = []
    num_hit_cache = 0
    with open(args.input_file) as fd:
        for line in fd:
            example = json.loads(line)
            if example["prompt"] in cache_prompts:
                num_hit_cache += 1
                continue
            examples.append(example)
    assert len(examples) != 0
    assert "prompt" in examples[0]

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)
    
    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start:args.end]

    total_examples = len(examples)
    return examples

@timeout_decorator.timeout(5)
def extract_majority_answer(generations):
    try:
        """Extract the majority answer from a list of generations using math_verify for comparison"""
        answers = []
        
        # Extract answers from each generation
        for gen in generations:
            try:
                extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
                parsed_answer = parse(gen, extraction_config=extraction_target)
                answers.append(parsed_answer)
            except Exception as e:
                print(f"Error extracting answer: {e}")
                continue
        
        if not answers:
            return None
            
        # Group answers by equivalence (using math_verify.verify)
        groups = []
        for answer in answers:
            # Skip None answers
            if answer is None:
                continue
                
            found_group = False
            for group in groups:
                if verify(group[0], answer):
                    group.append(answer)
                    found_group = True
                    break
            
            if not found_group:
                groups.append([answer])
        
        # Sort groups by size (largest first)
        groups.sort(key=len, reverse=True)
        
        # Return the most common answer if there are any groups
        if groups:
            return str(groups[0][0][-1])
        return None
    except:
        return None


def main(args, dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                              GPUs_per_dp_rank))
                              
    examples = prepare_data(args)
    print(f"DP rank {dp_rank}. full sample size: {len(examples)}")

    # Calculate basic number of samples per rank and remainder
    base_size = len(examples) // dp_size
    extras = len(examples) % dp_size

    # Calculate start and end positions for current rank
    # First extras ranks process one extra sample
    start = dp_rank * base_size + min(dp_rank, extras)
    end = start + base_size + (1 if dp_rank < extras else 0)

    examples = examples[start:end]
    print(f"DP rank {dp_rank} really needs to process sample size: {len(examples)}")
    if dp_rank == 0 and examples:
        print(examples[0])
        
    # Init python executor
    executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=args.func_call_timeout)

    # Load model if there are examples to process
    if len(examples) > 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        llm = LLM(model=args.model_name_or_path, max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True)
    
    # Prepare samples with proper prompts
    samples = []
    for example in examples:
        initial_prompt = example["prompt"]
        messages = [
            {"role": "user", "content": initial_prompt}, 
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        example["prompt"] = prompt
        samples.append(example)

    # print("input_file:", args.input_file, "samples:", len(samples))
    if (dp_rank == 0) and (len(samples) > 0):
        print("-" * 50)
        print("sample:", samples[0]['prompt'])
        print("-" * 50)

    # Prepare all prompts for batch processing - creating n_sampling copies of each prompt
    remain_prompts = []
    idx = 0
    for sample in samples:
        for _ in range(args.n_sampling):
            # remain_prompts.append((idx, sample['prompt']))
            # add cumulative_logprob, cumulative_seq_len
            remain_prompts.append((idx, sample['prompt'], 0, 0))
            idx += 1

    # Start timing
    start_time = time.time()
    max_func_call = args.max_func_call
    end_prompts = []

    # Main inference loop (original ToRA approach)
    for epoch in range(max_func_call):
        if epoch != (max_func_call - 1):
            stop_tokens = args.stop
        else:
            stop_tokens = []

        if dp_rank == 0:
            print("=" * 50, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        outputs = llm.generate(prompts, SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        n=1,
                        stop=stop_tokens,
                        logprobs=True, 
        ))
        
        # Process outputs
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs_t = []
        outputs_cumulative_logprob = []
        outputs_cumulative_seq_len = []
        for output in outputs:
            output_text = output.outputs[0].text
            output_finish_reason = output.outputs[0].finish_reason
            output_stop_reason = output.outputs[0].stop_reason
            output_cumulative_logprob = output.outputs[0].cumulative_logprob
            output_cumulative_seq_len = len(output.outputs[0].token_ids)
            try:
                if (output_finish_reason == "stop") and (output_stop_reason is not None):
                    if output_stop_reason not in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                        output_text += output_stop_reason
            except Exception as e:
                print(f"error: {e}")
                print(f"output_finish_reason: {output_finish_reason}, type(output_finish_reason): {type(output_finish_reason)}")
                print(f"output_stop_reason: {output_stop_reason}, type(output_stop_reason): {type(output_stop_reason)}")
                raise
            outputs_t.append(output_text)
            outputs_cumulative_logprob.append(output_cumulative_logprob)
            outputs_cumulative_seq_len.append(output_cumulative_seq_len)
        outputs = outputs_t
        assert len(outputs) == len(current_prompts)

        # Process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query, old_cumulative_logprob, old_cumulative_seq_len), output, output_cumulative_logprob, output_cumulative_seq_len in zip(current_prompts, outputs, outputs_cumulative_logprob, outputs_cumulative_seq_len):
            query += output
            new_cumulative_logprob = old_cumulative_logprob + output_cumulative_logprob
            new_cumulative_seq_len = old_cumulative_seq_len + output_cumulative_seq_len
            # if ("```python" in output) and output.endswith(args.stop[0]):
            #     if args.func_call_mode == "jupyter":
            #         program = extract_jupyter_like_program(query)
            #     else:
            #         program = extract_program(query)
            #     remain_prompts.append((i, query, new_cumulative_logprob, new_cumulative_seq_len))
            #     remain_codes.append(program)
            # else:
            #     end_prompts.append((i, query, new_cumulative_logprob, new_cumulative_seq_len))
            if output.endswith(args.stop[0]):
                if "```python" in output:
                    if args.func_call_mode == "jupyter":
                        program = extract_jupyter_like_program(query)
                    else:
                        program = extract_program(query)
                    remain_prompts.append((i, query, new_cumulative_logprob, new_cumulative_seq_len))
                    remain_codes.append(program)
                else:
                    remain_prompts.append((i, query, new_cumulative_logprob, new_cumulative_seq_len))
                    remain_codes.append("")
            else:
                end_prompts.append((i, query, new_cumulative_logprob, new_cumulative_seq_len))

        # Execute the remain prompts
        if remain_codes:
            remain_results = executor.batch_apply(remain_codes)
            for k in range(len(remain_prompts)):
                i, query, cumulative_logprob, cumulative_seq_len = remain_prompts[k]
                res, report = remain_results[k]
                if res == "empty code":
                    remain_prompts[k] = (i, query, cumulative_logprob, cumulative_seq_len)
                else:
                    exec_result = res if res else report
                    # run out of python exe
                    if epoch == max_func_call - 2:
                        exec_result += f"\n\n[SYSTEM]\nYou have exceeded the allowed number of code executions. You can no longer write or run code. Please continue solving the problem using your reasoning and analytical skills."
                    exec_result = f"```output\n{exec_result}\n```\n"
                    query += exec_result
                    # # not end
                    # if epoch == max_func_call - 1:
                    #     query += f"```output\n[SYSTEM] Maximum allowed function calls reached. Python runtime has been terminated. Further code execution is permanently disabled.\n```\n"
                    remain_prompts[k] = (i, query, cumulative_logprob, cumulative_seq_len)

    # Add any unsolved samples to end_prompts
    end_prompts.extend(remain_prompts)
    
    # Sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    time_use = time.time() - start_time
    print(f"DP rank {dp_rank}. Time use: {time_use:.2f}s")
    avg_time_use = time_use / len(samples)

    # Reorganize results back to samples
    all_samples = []
    for i, sample in enumerate(samples):
        initial_prompt = sample["prompt"]
        initial_prompt_len = len(initial_prompt)
        # Get the n_sampling generations for this sample
        sample_queries = end_prompts[i*args.n_sampling: (i+1)*args.n_sampling]
        generations = []
        generations_cumulative_logprob = []
        generations_cumulative_seq_len = []
        for j, query, cumulative_logprob, cumulative_seq_len in sample_queries:
            generations.append(query[initial_prompt_len:])
            generations_cumulative_logprob.append(cumulative_logprob)
            generations_cumulative_seq_len.append(cumulative_seq_len)
        
        sample["gen"] = generations
        sample["gen_cumulative_logprob"] = generations_cumulative_logprob
        sample["gen_cumulative_seq_len"] = generations_cumulative_seq_len
        
        # Extract and add majority answer if requested
        if args.extract_majority_answer:
            sample["majority_answer"] = extract_majority_answer(generations)
        
        sample["time_use"] = avg_time_use

        all_samples.append(sample)
    print(f"DP rank {dp_rank}. len(all_samples): {len(all_samples)}")
    
    # Write results to file
    output_file = os.path.join(args.output_dir, f"dp_rank_{dp_rank}.jsonl")
    while os.path.exists(output_file):
        output_file = output_file.replace(".jsonl", ".more.jsonl")
    with open(output_file, "w", encoding='utf-8') as fw:
        for sample in all_samples:
            try:
                dump = json.dumps(sample, ensure_ascii=False)
                fw.write(dump + "\n")
            except Exception as e:
                print(f"write error: {e}; skip!!!")
                continue

    print(f"Write inference results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    DP_size = args.data_parallel_size
    GPUs_per_dp_rank = args.tensor_parallel_size
    for i in range(DP_size):
        proc = Process(target=main,
                       args=(args, DP_size, i, dp_master_ip, dp_master_port,
                             GPUs_per_dp_rank))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()