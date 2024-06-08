#!/bin/bash

# On AWS, the EFA and OFI paths enable NCCL to use optimized networking.
#export LD_LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH

export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

MODEL_ARGS="\
--model_path=/gpfs/llama3/hf/70b_instruction_tuned
--model_arch=embedllama
--model_variant=70b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama3-70b-tp/
--ckpt_save_path=/gpfs/suneja/checkpoints/llama3-70b-tp/
--logical_shards=768
--sharding_strategy=tp
--seq_length=8192
--batch_size=1
--report_interval=10
--checkpoint_interval=5000
--num_steps=21000
--stage2_start_step=15000
--stage2_batch_size=36
--n_speculator_heads=4
--speculator_width=8192
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--data_path=/gpfs/suneja/datasets/llama3-dolma
--datasets="'dataset=stack'"
--weights="'1'"
"
#--data_path=/gpfs/llama3-common-crawl/rel0_7/lang=en
#--datasets="'dataset=commoncrawl'"

MODEL_ARGSX="\
--model_path=/gpfs/llama3/hf/8b_instruction_tuned
--model_arch=embedllama
--model_variant=8b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama3-8b-tmp
--ckpt_save_path=/gpfs/suneja/checkpoints/llama3-8b-tmp
--data_path=/gpfs/llama3-common-crawl/rel0_7/lang=en
--datasets="'dataset=commoncrawl'"
--logical_shards=768
--sharding_strategy=hsdp
--seq_length=8192
--batch_size=1
--report_interval=10
--checkpoint_interval=5000
--num_steps=10360
--stage2_start_step=150
--stage2_batch_size=36
--n_speculator_heads=4
--speculator_width=3584
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--weights="'1'"
"
#--data_path=/gpfs/suneja/datasets/llama3-dolma
#--datasets="'dataset=stack'"

MODEL_ARGS0="\
--model_path=/gpfs/llama3/hf/8b_instruction_tuned
--model_arch=embedllama
--model_variant=8b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama3-8b-ropefixed-scalednorm-tiewt-dolma-4heads-tp-multinode-simulated
--ckpt_save_path=/gpfs/suneja/checkpoints/llama3-8b-ropefixed-scalednorm-tiewt-dolma-4heads-tp-multinode-simulated
--data_path=/gpfs/suneja/datasets/llama3-dolma
--logical_shards=768
--sharding_strategy=tp
--seq_length=8192
--batch_size=1
--report_interval=10
--checkpoint_interval=20
--num_steps=15000
--stage2_start_step=20
--stage2_batch_size=36
--n_speculator_heads=4
--speculator_width=4096
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--datasets="'dataset=stack'"
--weights="'1'"
"
#--num_steps=15000

MODEL_ARGS1="\
--model_path=/gpfs/suneja/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/
--model_arch=embedllama
--model_variant=7b
--ckpt_load_path=/gpfs/suneja/checkpoints/llama2-7b-tmp-1
--ckpt_save_path=/gpfs/suneja/checkpoints/llama2-7b-tmp-1
--logical_shards=768
--sharding_strategy=hsdp
--seq_length=4096
--batch_size=8
--report_interval=10
--checkpoint_interval=5000
--num_steps=15000
--stage2_start_step=10000
--stage2_batch_size=96
--n_speculator_heads=3
--speculator_width=4096
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--data_path=/gpfs/suneja/datasets/bp7_llama2/lang=en
--datasets="'dataset=arxiv'"
--weights="'1'"
"
#--data_path=/gpfs/v7_high_quality_rerun_fuzzy_deduped/lang=en
#--datasets="'dataset=commoncrawl'"

MODEL_ARGS2="\
--model_path=/gpfs/prangan/granite-20b-code-instruct
--model_arch=embedgpt_bigcode
--model_variant=20b
--ckpt_load_path=/gpfs/suneja/checkpoints/granite-20b-tmp
--ckpt_save_path=/gpfs/suneja/checkpoints/granite-20b-tmp
--data_path=/gpfs/suneja/datasets/bluepile-granite
--logical_shards=768
--sharding_strategy=hsdp
--seq_length=8192
--batch_size=1
--report_interval=10
--checkpoint_interval=5000
--num_steps=5
--stage2_start_step=10000
--stage2_batch_size=24
--n_speculator_heads=3
--speculator_width=3072
--use_torch_compile=False
--learning_rate=1e-4
--seed=42
--datasets="'lang=en/dataset=github_clean'"
--weights="'1'"
"

MODEL_ARGS_GRANITE34B="\
--model_path=/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/
--model_arch=embedgpt_bigcode
--model_variant=34b
--ckpt_load_path=/gpfs/suneja/checkpoints/granite-34b-tp-tmp
--ckpt_save_path=/gpfs/suneja/checkpoints/granite-34b-tp-tmp
--data_path=/gpfs/suneja/datasets/bluepile-granite
--logical_shards=768
--sharding_strategy=tp
--seq_length=8192
--batch_size=2
--report_interval=10
--checkpoint_interval=5000
--num_steps=21000
--stage2_start_step=15000
--stage2_batch_size=48
--n_speculator_heads=5
--speculator_width=6144
--use_torch_compile=True
--learning_rate=1e-3
--seed=42
--datasets="'lang=en/dataset=github_clean'"
--weights="'1'"
"

MODEL_ARGS_GRANITE8B="\
--model_path=/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64/
--model_arch=embedcalico
--model_variant=8b
--ckpt_load_path=/gpfs/suneja/checkpoints/granite-8b-tmp
--ckpt_save_path=/gpfs/suneja/checkpoints/granite-8b-tmp
--data_path=/gpfs/suneja/datasets/bluepile-granite
--logical_shards=768
--sharding_strategy=None
--seq_length=4096
--batch_size=1
--report_interval=10
--checkpoint_interval=5000
--num_steps=15000
--stage2_start_step=30
--stage2_batch_size=24
--n_speculator_heads=5
--speculator_width=4096
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--datasets="'lang=en/dataset=github_clean'"
--weights="'1'"
"

MODEL_ARGS_GRANITE3B="\
--model_path=/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/
--model_arch=embedcalico
--model_variant=3b
--ckpt_load_path=/gpfs/suneja/checkpoints/granite-3b-tmp
--ckpt_save_path=/gpfs/suneja/checkpoints/granite-3b-tmp
--data_path=/gpfs/suneja/datasets/bluepile-granite
--logical_shards=768
--sharding_strategy=None
--seq_length=2048
--batch_size=1
--report_interval=10
--checkpoint_interval=5000
--num_steps=15000
--stage2_start_step=30
--stage2_batch_size=24
--n_speculator_heads=5
--speculator_width=4096
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--datasets="'lang=en/dataset=github_clean'"
--weights="'1'"
"

MODEL_ARGS_CODELLAMA34B="\
--model_path=/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d
--model_arch=embedllama
--model_variant=34b
--ckpt_load_path=/gpfs/suneja/checkpoints/codellama-34b/
--ckpt_save_path=/gpfs/suneja/checkpoints/codellama-34b/
--logical_shards=768
--sharding_strategy=fsdp
--seq_length=16384
--batch_size=1
--report_interval=10
--checkpoint_interval=5000
--num_steps=21000
--stage2_start_step=50
--stage2_batch_size=36
--n_speculator_heads=4
--speculator_width=8192
--use_torch_compile=True
--learning_rate=1e-3
--seed=42
--data_path=/gpfs/v7_high_quality_rerun_fuzzy_deduped
--datasets="'lang=en/dataset=github_clean'"
--weights="'1'"
"

#export TORCH_LOGS="dynamo,recompiles"
export CUDA_LAUNCH_BLOCKING=1
DO_BACKGROUND=0

if [ $DO_BACKGROUND -eq 1 ]
then
    FOUT=nohup-`date +%s`.out
    echo $FOUT

    nohup torchrun \
        --nproc_per_node=8 \
        speculator/train_speculator.py \
        ${MODEL_ARGS_GRANITE34B}\
        >$FOUT &
else
    torchrun \
        --nproc_per_node=8 \
        speculator/train_speculator.py \
        ${MODEL_ARGS_CODELLAMA34B}
fi        


