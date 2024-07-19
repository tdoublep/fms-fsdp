#!/bin/bash

SPECULATOR_ARGS_LLAMA_AWS="\
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

SPECULATOR_ARGS_LLAMA2_7B="\
--architecture=paged_llama
--variant=7b
--model_path="/gpfs/suneja/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"
--tokenizer="/gpfs/suneja/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/llama2-7b-tmp-1/checkpoints/step_11200_ckp.pth"
--prompt_len=64
--data_path="/gpfs1/users/suneja/datasets/bpv7_high_quality_rerun_fuzzy_deduped_incomplete/lang=en/"
--subdata="dataset=commoncrawl"
--n_predict=3
--n_candidates=5
--threshes=[6,4,3]
--seed=211
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

SPECULATOR_ARGS_GRANITE_8B_HF="\
--architecture=paged_llama
--variant=calico.8b.code
--model_path="/gpfs/suneja/models/dmf_models/granite-8b-code-instruct-20240509"
--tokenizer_path="/gpfs/suneja/models/dmf_models/granite-8b-code-instruct-20240509"
--speculator_path="/gpfs/suneja/models/granite-8b-code-instruct-accelerator"
--speculator_load_type=hf_remote
--model_source=hf
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/"
--subdata="lang=en/dataset=github_clean"
--threshes=[6,5,4,3,3]
--seed=211
"

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

SPECULATOR_ARGS_GRANITE_3B_HF="\
--architecture=paged_llama
--variant=calico.3b.code
--model_path="/gpfs/suneja/models/dmf_models/granite-3b-code-instruct-20240509"
--tokenizer_path="/gpfs/suneja/models/dmf_models/granite-3b-code-instruct-20240509"
--speculator_path="/gpfs/suneja/checkpoints/granite-3b/checkpoints/granite-3b-code-instruct/accelerator"
--speculator_load_type=hf_remote
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

SPECULATOR_ARGS_GRANITE_34B_HF="\
--architecture=paged_gpt_bigcode
--variant=ibm.34b
--model_path="/gpfs/suneja/models/dmf_models/granite-34b-code-instruct-20240506"
--tokenizer_path="/gpfs/suneja/models/dmf_models/granite-34b-code-instruct-20240506"
--model_source=hf
--speculator_path="/gpfs/suneja/models/granite-34b-code-instruct-accelerator"
--speculator_load_type=hf_remote
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/"
--subdata="lang=en/dataset=github_clean"
--n_predict=5
--n_candidates=5
--threshes=[6,5,4,3,3]
--seed=211
"
#--model_path="/data/suneja/models/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
#--tokenizer_path="/data/suneja/models/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/" 

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


SPECULATOR_ARGS_CODELLAMA_34B="\
--architecture=paged_llama
--variant=34b.code
--model_path="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--tokenizer_path="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/codellama-34b/checkpoints/step_21001_ckp.pth"
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bp7_llama2"
--subdata="lang=en/dataset=github_clean"
--n_predict=5
--n_candidates=5
--threshes=[6,5,4,3,3]
--seed=211
"

SPECULATOR_ARGS_GRANITE20B_COBOL="\
--architecture=paged_gpt_bigcode
--variant=ibm.20b.cobol
--model_path="/gpfs/prangan/granite20b-cobol"
--tokenizer_path="/gpfs/prangan/granite20b-cobol"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/granite-20b-cobol-1/checkpoints/step_3601_ckp.pth"
--prompt_len=64
--data_path="/gpfs/prangan/data_g20bc_tokenizer/code_data"
--subdata="dataset=ptv15_to_unsupervised"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"
#--speculator_path="/gpfs/suneja/checkpoints/grantite-20b-code-instruct-v1-speculator/step_42001_ckp.pth"
#--speculator_path="/gpfs/suneja/checkpoints/granite-20b-cobol-st1/checkpoints/step_3001_ckp.pth"


SPECULATOR_ARGS_GRANITE20B_COBOL_PTV18="\
--architecture=paged_gpt_bigcode
--variant=ibm.20b.cobol.ptv18
--model_path="/gpfs/suneja/models/granite-20b-cobol-ptv18-4-8k-8k-10Btok"
--tokenizer_path="/gpfs/suneja/models/granite-20b-cobol-ptv18-4-8k-8k-10Btok"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/granite-20b-cobol-1/checkpoints/step_3601_ckp.pth"
--prompt_len=64
--data_path="/gpfs/prangan/data_g20bc_ptv18/code_data"
--subdata="ptv18_to_supervised"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"
#--data_path="/gpfs/prangan/data_g20bc_tokenizer/code_data"
#--subdata="dataset=ptv15_to_supervised"
#--subdata="dataset=ptv15_to_unsupervised"


SPECULATOR_ARGS_LLAMA3_8B_HF="\
--architecture=paged_llama
--variant=llama3.8b
--model_path=/gpfs/suneja/models/dmf_models/llama-3-8b-instruct-20240418
--tokenizer_path=/gpfs/suneja/models/dmf_models/llama-3-8b-instruct-20240418
--model_source=hf
--speculator_path="/gpfs/suneja/models/llama3-8b-accelerator"
--speculator_load_type=hf_remote
--prompt_len=64
--data_path="/gpfs/suneja/datasets/llama3-common-crawl/rel0_7/lang=en"
--subdata="'dataset=commoncrawl'"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"

SPECULATOR_ARGS_GRANITE_20B_HF="\
--architecture=paged_gpt_bigcode
--variant=ibm.20b
--model_path="/home/zrltpa/models/granite-20b-code-instruct-20240506/"
--tokenizer_path="/home/zrltpa/models/granite-20b-code-instruct-20240506/"
--model_source=hf
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bluepile-granite/lang\=en/"
--subdata="dataset=github_clean"
--speculator_load_type=hf_remote
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
--speculator_path="/tmp/models--ibm-granite--granite-20b-code-instruct-accelerator/snapshots/ba7a42036002f21f7536df750c398966b0cb8ad9/"
"
#--distributed
#--speculator_path="/tmp/models--ibm-granite--granite-20b-code-instruct-accelerator/snapshots/ba7a42036002f21f7536df750c398966b0cb8ad9/"

SPECULATOR_ARGS_LLAMA2_13B_HF="\
--architecture=paged_llama
--variant=13b
--model_path=/gpfs/suneja/models/dmf_models/llama-2-13b-chat-20231109
--tokenizer_path=/gpfs/suneja/models/dmf_models/llama-2-13b-chat-20231109
--model_source=hf
--speculator_path="/gpfs/suneja/models/llama-13b-accelerator"
--speculator_load_type=hf_remote
--prompt_len=64
--data_path="/gpfs/suneja/datasets/bp7_llama2/lang=en"
--subdata="'dataset=PG19'"
--n_predict=3
--n_candidates=5
--threshes=[6,4,3]
--seed=211
"


SPECULATOR_ARGS_LLAMA2_70B="\
--architecture=paged_llama
--variant=70b
--model_path="/gpfs/suneja/models/Llama-2-70b-chat-hf"
--tokenizer_path="/gpfs/suneja/models/Llama-2-70b-chat-hf"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/llama2-70b-tp-wtinitfix/checkpoints/step_18838_ckp.pth"
--prompt_len=64
--data_path="/gpfs1/users/suneja/datasets/bpv7_high_quality_rerun_fuzzy_deduped_incomplete/lang=en/"
--subdata="'dataset=commoncrawl'"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"

SPECULATOR_ARGS_GRANITE_13B="\
--architecture=paged_gpt_bigcode
--variant=ibm.13b
--model_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--tokenizer_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--speculator_path="/gpfs/suneja/checkpoints/granite-13b-chat-v2.1/checkpoints/step_15001_ckp.pth"
--model_source=hf
--prompt_len=64
--data_path="/gpfs1/users/suneja/datasets/bluepile-processing/rel0_4/tokens_gpt_neox/"
--subdata="dataset=commoncrawl"
--seed=211
--n_predict=5
--n_candidates=5
--threshes=[6,5,4,3,3]
"
#--subdata="dataset=github_clean"


SPECULATOR_ARGS_GRANITE_13B_CCONLY="\
--architecture=paged_gpt_bigcode
--variant=ibm.13b
--model_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--tokenizer_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--speculator_path="/gpfs/suneja/checkpoints/granite-13b-chat-v2.1-cconly/checkpoints/step_17294_ckp.pth"
--model_source=hf
--prompt_len=64
--data_path="/gpfs1/users/suneja/datasets/bluepile-processing/rel0_4/tokens_gpt_neox/"
--subdata="dataset=github_clean"
--seed=211
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
"
#--subdata="dataset=commoncrawl"


SPECULATOR_ARGS_GRANITE_13B_CCONLY_HF="\
--architecture=paged_gpt_bigcode
--variant=ibm.13b
--model_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--tokenizer_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--speculator_path="/gpfs/suneja/checkpoints/granite-13b-chat-v2.1-cconly/checkpoints/accelerator"
--speculator_load_type=hf_remote
--model_source=hf
--prompt_len=64
--data_path="/gpfs1/users/suneja/datasets/bluepile-processing/rel0_4/tokens_gpt_neox/"
--subdata="dataset=commoncrawl"
--seed=211
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
"

SPECULATOR_ARGS_LLAMA3_70B_SPECU2_CONVERTED="\
--architecture=paged_llama
--variant=llama3.70b
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer_path="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-specu2-wtinitfix/checkpoints/step_14212_ckp_specu_v1.pth"
--prompt_len=64
--data_path="/gpfs"
--subdata="'fineweb-edu'"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"


SPECULATOR_ARGS_LLAMA3_70B_SPECU2_CONVERTED_HF="\
--architecture=paged_llama
--variant=llama3.70b
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer_path="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-specu2-wtinitfix/checkpoints/70b_instruction_tuned/accelerator"
--speculator_load_type=hf_remote
--prompt_len=64
--data_path="/gpfs"
--subdata="'fineweb-edu'"
--n_predict=4
--n_candidates=5
--threshes=[6,4,3,3]
--seed=211
"

SPECULATOR_ARGS_LLAMA2_7B_SPECUV1_TMP="\
--architecture=paged_llama
--variant=7b
--model_path="/gpfs/suneja/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"
--tokenizer="/gpfs/suneja/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"
--model_source=hf
--speculator_path="/gpfs/suneja/checkpoints/llama2-7b-specuv2-tmp/checkpoints/step_21_ckp_specu_v1.pth"
--prompt_len=64
--data_path="/gpfs1/users/suneja/datasets/bpv7_high_quality_rerun_fuzzy_deduped_incomplete/lang=en/"
--subdata="dataset=commoncrawl"
--n_candidates=1
--n_predict=3
--threshes=[6,4,3]
--seed=211
"
#--speculator_path="/gpfs/suneja/checkpoints/llama2-7b-specuv1-tmp/checkpoints/step_21_ckp.pth"


DO_BACKGROUND=0

if [ $DO_BACKGROUND -eq 1 ]
then
    nohup torchrun \
        --nproc_per_node=8 \
        speculator/benchmark_speculator_logical.py \
        ${SPECULATOR_ARGS_LLAMA3_70B_SPECU2_CONVERTED_HF}\
        > nohup.out &
else
    export CUDA_VISIBLE_DEVICES=7
    torchrun \
        --nproc_per_node=1 \
        speculator/benchmark_speculator_logical.py \
        ${SPECULATOR_ARGS_GRANITE_20B_HF}
fi
