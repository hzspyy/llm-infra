# SmolLM3-3B evaluation scripts

We use the [LightEval](https://github.com/huggingface/lighteval/) library to benchmark our models.

## Setup

Use conda/uv/venv with `python>=3.11`.

For reproducibility, we recommend fixed versions of the libraries:

```sh
pip install uv
uv venv smol3_venv --python 3.11 
source smol3_venv/bin/activate

GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements.txt
```

## Running the evaluations

All commands below were run on 2 x H100s with 80GB of memory each, using the `vllm` backend.

### SmolLM3-3B base model

```bash
MODEL_ARGS="model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,max_model_length=32768,max_num_batched_tokens=32768,generation_parameters={temperature:0},tensor_parallel_size=2,gpu_memory_utilization=0.7"
lighteval vllm \
    "$MODEL_ARGS" \
    "smollm3_base_test.txt" \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### SmolLM3-3B mid-trained model

This is a pure reasoning model, so no hybrid thinking:

```sh 
MODEL_ARGS="model_name=HuggingFaceTB/SmolLM3-3B-checkpoints,revision=it-mid-training,dtype=bfloat16,tensor_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
lighteval vllm "$MODEL_ARGS" "smollm3_instruct.txt" \
    --use-chat-template \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### SmolLM3-3B post-trained model

```sh
# Use /think or /no_think to enable or disable extended thinking
SYSTEM_PROMPT="/no_think" 
MODEL_ARGS="model_name=HuggingFaceTB/SmolLM3-3B,dtype=bfloat16,tensor_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
lighteval vllm "$MODEL_ARGS" "smollm3_instruct.txt" \
    --use-chat-template \
    --system-prompt "$SYSTEM_PROMPT" \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

> [!NOTE]
> BFCL is not yet supported by LightEval, so we used a [fork](https://github.com/huggingface/gorilla/tree/smollm3/berkeley-function-call-leaderboard) of the public repo, with a dedicated [parser](https://github.com/huggingface/gorilla/blob/smollm3/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/smollm3.py) for SmolLM3-3B.