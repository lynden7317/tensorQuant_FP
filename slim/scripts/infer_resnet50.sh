#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/resnetv1_50-model

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# Run evaluation.
export CUDA_VISIBLE_DEVICES=0
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --labels_offset=1 \
  --max_num_batches=10 \
  --batch_size=2 \
#  --intr_quantizer=nearest,16,8 \
#  --intr_quantize_layers=block \
#  --extr_quantizer=16,8 \
#  --extr_quantize_layers= 

unset CUDA_VISIBLE_DEVICES
