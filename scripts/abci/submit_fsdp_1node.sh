#!/bin/bash

#$-l rt_AF=1
#$-l h_rt=0:10:00
#$-j y
#$ -o outputs/a-node/
#$-cwd

#submit: qsub -g gcd50698 scripts/abci/submit_fsdp_1node.sh

source /etc/profile.d/modules.sh
module load python/3.11/3.11.2 cuda/12.0/12.0.0 cudnn/8.9/8.9.7
source myenv/bin/activate

python fam/llm/train.py --num_gpus 8 --batch_size 16 --per_gpu_batchsize 1 --check_val_every_n_epoch 1 --exp_name kotoba_voice_1.3B_debug_abci_reazon_small --max_epoch 1 --debug_data_dir /groups/gcd50698/reazon_data/reazon_small --debug --fsdp_strategy SHARD_GRAD_OP
