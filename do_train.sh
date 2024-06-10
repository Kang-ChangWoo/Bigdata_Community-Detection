#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python N3-train.py \
  --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/real_trainset" \
  --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
  --testset real