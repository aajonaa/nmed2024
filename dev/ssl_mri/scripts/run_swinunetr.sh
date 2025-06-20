#!/bin/bash

# conda activate adrd

ps=16
vs=128
bs=2  # Further reduced batch size to avoid CUDA OOM
heads=6
embed_dim=384
n_samples=369  # BraTS2020 has 369 training samples
dataset="BraTS2020_${n_samples}"
outdim=8192
# arch="vit_tiny"
# export LD_PRELOAD=tcmalloc.so:$LD_PRELOAD

# Set data path to BraTS2020 dataset
data_path="data/MRI/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Set checkpoint directory
ckptdir="results/brats2020_ssl_swinunetr"
mkdir -p ${ckptdir}

echo "Starting BraTS2020 SSL fine-tuning..."
echo "Data path: ${data_path}"
echo "Checkpoint dir: ${ckptdir}"
echo "Dataset: ${dataset}"
echo "Batch size: ${bs}"

#CUDA_VISIBLE_DEVICES=0
# Set memory optimization environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OMP_NUM_THREADS=1 NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --master_port=29759 \
            main_swinunetr.py --logdir ${ckptdir} --epochs 50 --num_steps=500 --data_path ${data_path} --batch_size ${bs} --num_workers 0 \
            --use_checkpoint --eval_num 100 --smartcache_dataset