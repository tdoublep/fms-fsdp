#!/bin/bash

SPECULATOR_ARGS_LLAMA="\
--architecture=paged_llama
--variant=7b
--model_path="/lustre/llama_weights/hf/7B-F/"
--tokenizer="/lustre/llama_weights/hf/7B-F/"
--model_source=hf
--speculator_path="/lustre/suneja/checkpoints/llama-7b-speculator/step_21001_ckp.pth"
--prompt_len=64
--data_path="/lustre/bluepile-processing/rel0_7/tokens/llama2/high_quality_rerun_fuzzy_deduped/"
--subdata="lang\=en/dataset\=webhose/"
"

SPECULATOR_ARGS_GRANITE="\
--architecture=paged_gpt_bigcode
--variant=ibm.20b
--model_path="/gpfs/prangan/granite-20b-code-instruct"
--tokenizer_path="/gpfs/prangan/granite-20b-code-instruct"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/grantite-20b-code-instruct-v1-speculator/step_42001_ckp.pth"
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/"
--subdata="lang=en/dataset=github_clean"
--n_predict=4
--n_candidates=1
--threshes=[6,4,3,3]
--seed=211
"
#--variant=ibm.20b
#--model_path="/gpfs/prangan/granite-20b-code-instruct"
#--tokenizer_path="/gpfs/prangan/granite-20b-code-instruct"
#--data_path="/lustre/dwertheimer/datasets/bluepile-granite/"
#--model_path="/gpfs/suneja/models/granite-20b-code-instruct-v1"
#--tokenizer_path="/gpfs/suneja/models/granite-20b-code-instruct-v1"

#--variant=ibm.34b
#--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
#--tokenizer_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"

export CUDA_VISIBLE_DEVICES=1

torchrun \
    --nproc_per_node=1 \
    speculator/benchmark_speculator_logical.py \
    ${SPECULATOR_ARGS_GRANITE}

