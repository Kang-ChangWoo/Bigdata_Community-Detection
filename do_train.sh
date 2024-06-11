#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python N3-train.py \
#   --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/graph_dataset_v2" \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
#   --testset TC1 \
#   --pth TC1


# CUDA_VISIBLE_DEVICES=4 python N3-train.py \
#   --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/graph_dataset_v2" \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
#   --testset TC1 \
#   --ablation only_mlp \
#   --sMLP_odim 20 \
#   --pth TC1_only_mlp

# CUDA_VISIBLE_DEVICES=5 python N3-train.py \
#   --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/graph_dataset_v2" \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
#   --testset TC1 \
#   --ablation only_gcn \
#   --GCN_odim 20 \
#   --pth TC1_only_gcn

CUDA_VISIBLE_DEVICES=3 python N3-train.py \
  --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/real_trainset_v2" \
  --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
  --testset real \
  --pth real_Full_model \
  --ablation full_model \

CUDA_VISIBLE_DEVICES=3 python N3-train.py \
  --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/real_trainset_v2" \
  --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
  --testset real \
  --ablation only_MLP \
  --sMLP_odim 20 \
  --pth real_only_MLP

CUDA_VISIBLE_DEVICES=3 python N3-train.py \
  --train_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/real_trainset_v2" \
  --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
  --testset real \
  --ablation only_GCN \
  --GCN_odim 20 \
  --pth real_only_GCN