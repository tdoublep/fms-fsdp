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
--variant=ibm.34b
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--tokenizer_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
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


SPECULATOR_ARGS_GRANITE_8B="\
--architecture=paged_llama
--variant=calico.8b.code
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64"
--tokenizer_path="/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64"
--model_source=hf
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/"
--subdata="lang=en/dataset=github_clean"
--n_predict=5
--n_candidates=5
--threshes=[6,5,4,3,3]
--seed=211
"
#--speculator_path="/gpfs/prangan/backup-ckptx/checkpoints/step_15001_ckp.pth"

#need paged_gpt_bigcode + 3b changes to fms-extras beanch
SPECULATOR_ARGS_GRANITE_3B="\
--architecture=paged_llama
--variant=calico.3b.code
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--tokenizer_path="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--speculator_path="/gpfs/prangan/ckpts/granite_3b_stage1/checkpoints/step_21001_ckp.pth"
--model_source=hf
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/"
--subdata="lang=en/dataset=github_clean"
--n_predict=5
--n_candidates=5
--threshes=[6,5,4,3,3]
--seed=211
"

SPECULATOR_ARGS_GRANITE_34B="\
--architecture=paged_gpt_bigcode
--variant=ibm.34b
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--tokenizer_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp-tmp/checkpoints/step_21001_ckp.pth"
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/"
--subdata="lang=en/dataset=github_clean"
--n_predict=5
--n_candidates=5
--threshes=[6,5,4,3,3]
--seed=211
"

SPECULATOR_ARGS_LLAMA3_70B="\
--architecture=paged_llama
--variant=llama3.70b
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer_path="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-ropefixed-tie_wt-scalednorm-4node-backup/checkpoints/step_11186_ckp.pth"
--prompt_len=64
--data_path="/gpfs/llama3-common-crawl/rel0_7/lang=en"
--subdata="'dataset=commoncrawl'"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"
#--data_path="/gpfs/suneja/datasets/llama3-dolma"
#--subdata="'dataset=stack'"

export CUDA_VISIBLE_DEVICES=1

#nohup torchrun \
torchrun \
    --nproc_per_node=1 \
    speculator/benchmark_speculator_logical.py \
    ${SPECULATOR_ARGS_GRANITE_34B}\
    #> nohup.out &

