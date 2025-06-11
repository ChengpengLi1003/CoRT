#!/bin/bash
set -ex

MODEL_PATH=$1
PROMPT_TEMPLATE=reason_step
SELECTED_TESTS=aime24,aime25,amc23,math500,olympiad_bench

START=0
END=-1

DATA_PARALLEL_SIZE=8
TENSOR_PARALLEL_SIZE=1

DEFAULT_TEMPERATURE=0.6
DEFAULT_N_SAMPLING=16
MAX_MODEL_LEN=32768
MAX_FUNC_CALL=15
MAX_TOKENS_PER_CALL=32768

DEFAULT_TEST_SETS=("aime24" "aime25" "amc23" "math500" "olympiad_bench")

if [ ! -z "$SELECTED_TESTS" ]; then
    IFS=',' read -ra TEST_SETS <<< "$SELECTED_TESTS"
else
    TEST_SETS=("${DEFAULT_TEST_SETS[@]}")
fi

# Loop through each test set
for TEST_SET in "${TEST_SETS[@]}"; do
    echo "Processing test set: $TEST_SET"
    
    # Set different parameters for math500 and olympiad_bench
    if [[ "$TEST_SET" == "math500" ]] || [[ "$TEST_SET" == "olympiad_bench" ]]; then
        TEMPERATURE=0.6
        N_SAMPLING=4
    else
        TEMPERATURE=$DEFAULT_TEMPERATURE
        N_SAMPLING=$DEFAULT_N_SAMPLING
    fi
    
    OUTPUT_DIR=$MODEL_PATH/evaluation_outputs_dp_mj.temp_$TEMPERATURE.n_sampling_$N_SAMPLING.max_model_len_$MAX_MODEL_LEN.max_func_call_$MAX_FUNC_CALL.max_tokens_per_call_$MAX_TOKENS_PER_CALL/infer_on_test_$TEST_SET.$PROMPT_TEMPLATE.start_$START.end_$END
    mkdir -p $OUTPUT_DIR

    TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 python -m infer.inference_vllm_dp_mj \
        --input_file data/test_$TEST_SET.$PROMPT_TEMPLATE.jsonl \
        --start $START \
        --end $END \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL_PATH \
        --engine vllm \
        --temperature $TEMPERATURE \
        --top_p 0.95 \
        --n_sampling $N_SAMPLING \
        --stop_tokens_mode normal_code_block_end \
        --max_tokens_per_call $MAX_TOKENS_PER_CALL \
        --max_model_len $MAX_MODEL_LEN \
        --max_func_call $MAX_FUNC_CALL \
        --func_call_mode jupyter \
        --data_parallel_size $DATA_PARALLEL_SIZE \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --enable_cache

    OUTPUT_METRICS_FILE=$OUTPUT_DIR/metrics.json
    python evaluation/compute_metrics_dp.py \
        --input_dir $OUTPUT_DIR \
        --output_file $OUTPUT_METRICS_FILE
done
