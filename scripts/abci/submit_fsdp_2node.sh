#!/bin/bash

#$-l rt_AF=2
#$-l h_rt=0:10:00
#$-j y
#$ -o outputs/a-node/
#$-cwd

#submit: qsub -g gcd50698 scripts/abci/submit_fsdp_2node.sh

# module
source /etc/profile.d/modules.sh
module load python/3.11/3.11.2 cuda/12.0/12.0.0 cudnn/8.9/8.9.7 hpcx/2.12
source myenv/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
export NUM_GPU_PER_NODE=8
NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
    echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

mpirun -np $NUM_GPUS \
    -hostfile $HOSTFILE_NAME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -bind-to none -map-by slot \
    -x PATH \
    python fam/llm/train.py \
    --num_nodes 2 \
    --num_gpus 8 \
    --batch_size 16 \
    --per_gpu_batchsize 1 \
    --check_val_every_n_epoch 1 \
    --exp_name kotoba_voice_1.3B_debug_abci_reazon_small \
    --max_epoch 1 \
    --debug_data_dir /groups/gcd50698/reazon_data/reazon_small \
    --debug \
    --fsdp_strategy SHARD_GRAD_OP

# python -m torch.distributed.run \
#     --nnodes 2 \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --nproc_per_node 8 \
#     fam/llm/train.py \
#     --num_nodes 2 \
#     --num_gpus 8 \
#     --batch_size 16 \
#     --per_gpu_batchsize 1 \
#     --check_val_every_n_epoch 1 \
#     --exp_name kotoba_voice_1.3B_debug_abci_reazon_small \
#     --max_epoch 1 \
#     --debug_data_dir /groups/gcd50698/reazon_data/reazon_small \
#     --debug \
#     --fsdp_strategy SHARD_GRAD_OP
