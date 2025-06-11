import random
import os

import json
import argparse

import json
import datasets
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
from math_verify import  verify
from transformers import AutoTokenizer
from functools import partial

from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="gsm8k.jsonl", type=str)
    parser.add_argument("--tokenizer_name_or_path", default="path_to_DeepSeek-R1-Distill-Qwen-1.5B", type=str)
    parser.add_argument("--output_file", default="output.jsonl", type=str)
    args = parser.parse_args()
    return args

def calculate_sequence_entropy(cumulative_logprob, sequence_length):
    # 平均对数概率
    avg_logprob = cumulative_logprob / sequence_length
    # 转换为概率
    avg_prob = np.exp(avg_logprob)
    # 计算熵
    entropy = -avg_prob * avg_logprob
    return entropy


def process_one_sample(tokenizer, sample):
    try:
        gt_answer = str(sample["gt_answer"]).strip()
        latex_gt_answer = f"${gt_answer}$" if not gt_answer.startswith("$") else gt_answer
        parsed_gt_answer = parse(latex_gt_answer)

        truth = str(sample["gt_answer"])
        if "\\boxed" in truth:
            deepscaler_gt_answer = extract_answer(truth)
        else:
            deepscaler_gt_answer = truth
        
        if "gen" in sample:
            gens = sample["gen"]
        else:
            raise
        assert len(gens) >= 1

        # 解析每个生成的答案并验证正确性
        correctness_math_verify = []
        correctness_deepscaler = []
        parsed_answers = []
        gens_cumulative_logprob = sample.get("gen_cumulative_logprob", [])
        gens_cumulative_seq_len = sample.get("gen_cumulative_seq_len", [])
        gens_entropy = []
        gen_lens = []
        for i in range(len(gens)):
            gen = gens[i]

            extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
            parsed_boxed_answer = parse(gen, extraction_config=extraction_target)
            parsed_answers.append(parsed_boxed_answer)
            correctness = verify(parsed_gt_answer, parsed_boxed_answer)
            correctness_math_verify.append(correctness)

            if "</think>" in gen:
                model_solution = gen.split("</think>")[1]
            else:
                model_solution = ""
            deepscaler_model_answer = extract_answer(model_solution)
            if deepscaler_model_answer is None:
                deepscaler_model_answer = ""
                if correctness:
                    print("*" * 20 + "gen" + "*" * 20)
                    print(gen[:100] + "\n\n...\n\n" + gen[-1000:])
                    print("*" * 20 + "model_solution" + "*" * 20)
                    print(model_solution)
                    print("*" * 20 + "deepscaler_model_answer" + "*" * 20)
                    print(deepscaler_model_answer)
                    print("*" * 20 + "parsed_boxed_answer" + "*" * 20)
                    print(parsed_boxed_answer)
                    print("*" * 20 + "parsed_gt_answer" + "*" * 20)
                    print(parsed_gt_answer)
                    print("=" * 80)
            deepscaler_correctness = grade_answer_mathd(deepscaler_model_answer, deepscaler_gt_answer) or grade_answer_sympy(deepscaler_model_answer, deepscaler_gt_answer)
            correctness_deepscaler.append(deepscaler_correctness)

            gen_lens.append(len(tokenizer.encode(gen, add_special_tokens=False)))
            if (len(gens_cumulative_logprob) == len(gens)) and (len(gens_cumulative_seq_len) == len(gens)):
                gen_cumulative_logprob = gens_cumulative_logprob[i]
                gen_cumulative_seq_len = gens_cumulative_seq_len[i]
                gen_entropy = calculate_sequence_entropy(gen_cumulative_logprob, gen_cumulative_seq_len)
            else:
                gen_entropy = -1
            gens_entropy.append(gen_entropy)
        
        if "majority_answer" in sample:
            mj_answer = sample["majority_answer"]
            latex_mj_answer = f"${str(mj_answer)}$" if not str(mj_answer).startswith("$") else str(mj_answer)
            parsed_mj_answer = parse(latex_mj_answer)
            majority_answer_correctness = verify(parsed_gt_answer, parsed_mj_answer)
            sample["majority_answer_correctness"] = majority_answer_correctness
        else:
            # 如果没有majority_answer，基于parsed_boxed_answer投票计算majority_answer
            # 使用verify函数进行两两比对，找出相同的答案
            answer_groups = []
            for parsed_answer in parsed_answers:
                # 检查当前答案是否与已有的组匹配
                found_group = False
                for group in answer_groups:
                    # 检查与组的第一个元素是否相同
                    if verify(group[0], parsed_answer):
                        group.append(parsed_answer)
                        found_group = True
                        break
                
                # 如果没有找到匹配的组，创建新组
                if not found_group:
                    answer_groups.append([parsed_answer])

            # 找出最大的组（出现次数最多的答案）
            majority_group = max(answer_groups, key=len)
            majority_parsed_answer = majority_group[0]  # 使用组中的第一个答案作为代表

            # 验证majority_answer的正确性
            majority_answer_correctness = verify(parsed_gt_answer, majority_parsed_answer)
            majority_answer = majority_parsed_answer[-1]
            sample["majority_answer"] = majority_answer
            sample["majority_answer_correctness"] = majority_answer_correctness

        sample["correctness_math_verify"] = correctness_math_verify
        sample["correctness_deepscaler"] = correctness_deepscaler
        sample["gen_lens"] = gen_lens
        if gens_entropy != []:
            sample["gen_entropy"] = gens_entropy
            sample["gen_entropy_mean"] = np.mean(gens_entropy)
            sample["gen_entropy_std"] = np.std(gens_entropy)
            sample["gen_entropy_cv"] = np.std(gens_entropy) / np.mean(gens_entropy)
            sample["gen_entropy_range"] = np.ptp(gens_entropy)
            sample["gen_entropy_percentiles"] = np.percentile(gens_entropy, [25, 50, 75])
        correctness_count = sum(correctness_math_verify)
        correctness_deepscaler_count = sum(correctness_deepscaler)
        if correctness_count > 0:
            sample["correctness_at_least_one_pass"] = 1
        else:
            sample["correctness_at_least_one_pass"] = 0
        if correctness_deepscaler_count > 0:
            sample["correctness_deepscaler_at_least_one_pass"] = 1
        else:
            sample["correctness_deepscaler_at_least_one_pass"] = 0
        sample["correctness_avg_pass_rate"] = correctness_count / len(correctness_math_verify)
        sample["correctness_deepscaler_avg_pass_rate"] = correctness_deepscaler_count / len(correctness_deepscaler)
        return sample
    except Exception as e:
        if "gen" in sample:
            gens = sample["gen"]
        gens_count = len(gens)
        sample["correctness_math_verify"] = [False] * gens_count
        sample["correctness_deepscaler"] = [False] * gens_count
        sample["gen_lens"] = [-1] * gens_count
        sample["correctness_at_least_one_pass"] = 0
        sample["correctness_avg_pass_rate"] = 0 / len(sample["correctness_math_verify"])
        sample["correctness_deepscaler_avg_pass_rate"] = 0 / len(sample["correctness_deepscaler"])
        return sample

def main(args):
    samples = []
    touched_prompts = set()
    # with open(args.input_file) as fd:
    #     for line in fd:
    #         sample = json.loads(line)
    #         samples.append(sample)
    input_dir_ents = os.listdir(args.input_dir)
    for de in input_dir_ents:
        if not de.startswith("dp_rank"):
            continue
        input_file = os.path.join(args.input_dir, de)
        with open(input_file) as fd:
            for line in fd:
                sample = json.loads(line)
                if sample["prompt"] in touched_prompts:
                    continue
                samples.append(sample)
                touched_prompts.add(sample["prompt"])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # process_one_sample(tokenizer, samples[0])
    # raise

    timeout_cnt = 0
    num_verify_error = 0
    results = []
    partial_process = partial(process_one_sample, tokenizer)
    with ProcessPool(max_workers=8) as pool:
        future = pool.map(partial_process, samples, timeout=30)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    timeout_cnt += 1
                    num_verify_error += 1
                    continue
                except Exception as error:
                    num_verify_error += 1
                    print(error)
                    continue
                progress_bar.update(1)

    source_to_samples = {}
    for sample in results:
        source = sample.get("source", "NA")
        if source not in source_to_samples:
            source_to_samples[source] = []
        source_to_samples[source].append(sample)

    metrics = {}
    for source, source_samples in source_to_samples.items():
        pass1 = []
        avg = []
        mj = []

        pass1_deepscaler = []
        avg_deepscaler = []

        time_uses = []
        correct_response_lens = []
        wrong_response_lens = []
        overall_response_lens = []
        gens_entropy_mean = []
        gens_entropy_cv = []
        for sample in source_samples:
            pass1.append(sample["correctness_at_least_one_pass"])
            avg.append(sample["correctness_avg_pass_rate"])
            
            pass1_deepscaler.append(sample["correctness_deepscaler_at_least_one_pass"])
            avg_deepscaler.append(sample["correctness_deepscaler_avg_pass_rate"])

            if "majority_answer_correctness" in sample:
                mj.append(sample["majority_answer_correctness"])
            if "time_use" in sample:
                time_uses.append(sample["time_use"])
            if "gen_entropy_mean" in sample:
                gens_entropy_mean.append(sample["gen_entropy_mean"])
                gens_entropy_cv.append(sample["gen_entropy_cv"])
            for correctness, gen_len in zip(sample["correctness_math_verify"], sample["gen_lens"]):
                if correctness:
                    correct_response_lens.append(gen_len)
                else:
                    wrong_response_lens.append(gen_len)
                overall_response_lens.append(gen_len)

        k = len(source_samples[0]["gen"])
        source_metrics = {
            f"pass@{k}": sum(pass1) / len(pass1), 
            f"avg@{k}": sum(avg) / len(avg), 
            f"mj@{k}": sum(mj) / len(mj) if len(mj) > 0 else -1, 
            f"pass@{k}_deepscaler": sum(pass1_deepscaler) / len(pass1_deepscaler), 
            f"avg@{k}_deepscaler": sum(avg_deepscaler) / len(avg_deepscaler), 
            f"avg_tokens_correct_response": np.mean(correct_response_lens) if len(correct_response_lens) > 0 else -1, 
            f"avg_tokens_incorrect_response": np.mean(wrong_response_lens) if len(wrong_response_lens) > 0 else -1, 
            f"avg_tokens_overall_response": np.mean(overall_response_lens) if len(overall_response_lens) > 0 else -1, 
            f"avg_entropy_mean": np.mean(gens_entropy_mean) if len(gens_entropy_mean) > 0 else -1, 
            f"avg_entropy_cv": np.mean(gens_entropy_cv) if len(gens_entropy_cv) > 0 else -1, 
            f"time_use_mean": np.mean(time_uses) if len(time_uses) > 0 else -1, 
            f"time_use_std": np.std(time_uses) if len(time_uses) > 0 else -1, 
            f"time_use_sum": np.sum(time_uses) if len(time_uses) > 0 else -1, 
            "sample_size": len(source_samples), 
            "num_verify_error": num_verify_error, 
        }
        metrics[source] = source_metrics

    metrics_json = json.dumps(metrics, indent=4)
    print("Evaluation Metrics:")
    print(metrics_json)
    with open(args.output_file, "w") as fw:
        fw.write(metrics_json)
    print(f"write to {args.output_file}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)