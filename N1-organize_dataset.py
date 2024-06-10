import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse

from evaluation import eval, calculate_nmi, load_ground_truth
from sklearn.metrics.cluster import normalized_mutual_info_score

import os, shutil
import fnmatch
from tqdm import tqdm

# test set = 'real' or 'TC1'
def move_and_rename_dat_files(path, testset='real'):
    subdirectories = [d for d in os.listdir(path) if not d.endswith('.DS_Store')]
    print(path)
    print(subdirectories)

    if testset == 'real':
        for subdirectory in subdirectories:
            sub_path = os.path.join(path, subdirectory)

            items = os.listdir(sub_path)

            for item in items:
                item_path = os.path.join(sub_path, item)

                if item.endswith('.dat'):
                    new_name = f"{subdirectory}-{item}"
                    new_path = os.path.join(path, new_name)
                    shutil.move(item_path, new_path)
                    print(f"Moved {item} to {new_path}")
            
            shutil.rmtree(sub_path)
            print(f"Deleted directory {sub_path}")


    elif testset == 'TC1':
        for subdirectory in subdirectories:
            sub_path = os.path.join(path, subdirectory)
            items = os.listdir(sub_path)

            for item in items:
                item_path = os.path.join(sub_path, item)
                if item.endswith('.dat'):
                    new_name = f"{item}"
                    new_path = os.path.join(path, new_name)
                    shutil.move(item_path, new_path)
                    print(f"Moved {item} to {new_path}")
            
            shutil.rmtree(sub_path)
            print(f"Deleted directory {sub_path}")

def main(args):
    if args.isit_hierarchy == 1:
        move_and_rename_dat_files(args.dataset_path, args.testset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT', help='-')
    parser.add_argument('--isit_hierarchy', type=int, default=1, help='-')
    parser.add_argument('--testset', type=str, default='TC1', help='-')
    args = parser.parse_args()
    main(args)