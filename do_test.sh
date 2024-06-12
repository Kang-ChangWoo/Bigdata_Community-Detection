#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python N4-test.py \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
#   --model_path "./ckpt/real_only_GCN-last.pt" \
#   --testset real \
#   --ablation only_GCN \
#   --GCN_odim 20

  
# CUDA_VISIBLE_DEVICES=3 python N4-test.py \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
#   --model_path "./ckpt/real_only_MLP-last.pt" \
#   --testset real \
#   --ablation only_MLP \
#   --sMLP_odim 20


  # CUDA_VISIBLE_DEVICES=3 python N4-test.py \
  # --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
  # --model_path "./ckpt/real_Full_model-last.pt" \
  # --testset real \
  # --ablation full_model 

  #   CUDA_VISIBLE_DEVICES=3 python N4-test.py \
  # --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
  # --model_path "./ckpt/real_Full_model-last.pt" \
  # --testset real \
  # --ablation full_model 

# CUDA_VISIBLE_DEVICES=3 python N4-test.py \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/real-world_dataset" \
#   --model_path "./ckpt/real_only_MLP-last.pt" \
#   --ablation full_model \
#   --testset real

CUDA_VISIBLE_DEVICES=3 python N4-test.py \
  --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
  --model_path "./ckpt/TC1_only_GCN-last.pt" \
  --ablation only_GCN \
  --testset TC1

# CUDA_VISIBLE_DEVICES=3 python N4-test.py \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
#   --model_path "./ckpt/TC1_only_MLP-last.pt" \
#   --ablation only_MLP \
#   --testset TC1

# CUDA_VISIBLE_DEVICES=3 python N4-test.py \
#   --test_path "/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT" \
#   --model_path "./ckpt/TC1-last.pt" \
#   --ablation full_model \
#   --testset TC1


# python N4-test.py                                                  \         
#   --test_path 'path/to/your/`TC1 dataset`'                         \         
#   --model_path 'path/to/your/model_checkpoints'                    \         
#   --testset TC1                                                    