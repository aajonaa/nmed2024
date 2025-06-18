#!/bin/bash

conda activate adrd

ps=16
vs=128
bs=4  # Reduced batch size for TotalSegmentator dataset
heads=6
embed_dim=384
n_samples=298  # TotalSegmentator has 298 samples
dataset="TotalSegmentator_${n_samples}"
outdim=8192
# arch="vit_tiny"
# export LD_PRELOAD=tcmalloc.so:$LD_PRELOAD

# Set data path to TotalSegmentator dataset
data_path="./data/MRI/TotalsegmentatorMRI_dataset_v100"

# Set checkpoint directory
ckptdir="results/totalsegmentator_ssl_swinunetr"
mkdir -p ${ckptdir}

echo "Starting TotalSegmentator SSL pre-training..."
echo "Data path: ${data_path}"
echo "Checkpoint dir: ${ckptdir}"
echo "Dataset: ${dataset}"
echo "Batch size: ${bs}"

#CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=1 NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port=29759 \
            main_swinunetr.py --logdir ${ckptdir} --epochs 50 --num_steps=50000 --data_path ${data_path} --batch_size ${bs} --num_workers 3 \
            --use_checkpoint --eval_num 100 --cache_dataset