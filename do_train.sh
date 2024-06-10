#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py \
  --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/graph_dataset_v2" \
  --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
  --testset TC1