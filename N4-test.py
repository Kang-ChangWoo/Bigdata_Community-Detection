import os, argparse, random, glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

import fnmatch

from natsort import natsorted
import os
import networkx as nx
import igraph as ig 
import leidenalg


from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from torch.utils.data import DataLoader  


from utils.utils import load_network, load_ground_truth
from utils.utils import extract_meta_info, calculate_nmi
from utils.dataloader import train_graph_loader, test_graph_loader
from utils.detection import detect_communities_louvain, detect_communities_leiden
from utils.network import mainnet_ResolNet, onlyMLP_ResolNet, onlyGCN_ResolNet




def save_results_to_txt(results, file_name):
    with open(file_name, 'w') as f:
        for result in results:
            f.write(f"{result[0]} {result[1]:.6f} {result[2]:.6f}\n")


"""
3. Neural network tool-kit.
   (Dataloader, Network model)
"""
class graph_loader(torch.utils.data.Dataset):
    def __init__(self, args, file_path):
        self.file_path = file_path
        
        self.graphs = glob.glob(os.path.join(file_path, '*-*.dat' ))

        self.graphs = [i for i in self.graphs if not fnmatch.fnmatch(i, '*-c.dat')]
        self.graphs = [i for i in self.graphs if i.endswith('dat')]
        
        self.graphs = natsorted(self.graphs)

        self.community_labels = glob.glob(os.path.join(file_path, '*-*-c.dat' ))
        self.community_labels = natsorted(self.community_labels)

        self.community_labels = [i for i in self.community_labels if i.endswith('dat')]

        self.args = args

        self.peak_resolutions = np.loadtxt(os.path.join(file_path,'label.txt'), dtype=str)

    def __getitem__(self, idx):
        G = load_network(self.graphs[idx])
        torch_G = from_networkx(G).to(self.args.device)
        torch_G.x = torch.ones((torch_G.num_nodes, 1), dtype=torch.float).to(self.args.device)

        community_label = self.community_labels[idx]
        _, nmi_score, peak_resolution = self.peak_resolutions[idx]


        nmi_score = torch.tensor([float(nmi_score)]).to(self.args.device)
        peak_resolution = torch.tensor([float(peak_resolution)]).to(self.args.device)
        return G, torch_G, community_label, peak_resolution, nmi_score
        
    def __len__(self):
        return len(self.graphs)
        

class test_graph_loader(torch.utils.data.Dataset):
    def __init__(self, args, test_path):
        self.test_path = test_path
        self.args = args

        if args.testset == 'real':
            self.test_graphs = glob.glob(os.path.join(test_path, '*-network.dat' ))
            self.test_graphs = natsorted(self.test_graphs)

            self.test_community_labels = glob.glob(os.path.join(test_path, '*-community.dat' ))
            self.test_community_labels = natsorted(self.test_community_labels)  

        else:
            self.test_graphs = glob.glob(os.path.join(test_path, '*-*.dat' ))
            self.test_graphs = [i for i in self.test_graphs if not fnmatch.fnmatch(i, '*-c.dat')]
            self.test_graphs = [i for i in self.test_graphs if i.endswith('dat')]
            self.test_graphs = natsorted(self.test_graphs)

            self.test_community_labels = glob.glob(os.path.join(test_path, '*-*-c.dat' ))
            self.test_community_labels = [i for i in self.test_community_labels if i.endswith('dat')]
            self.test_community_labels = natsorted(self.test_community_labels) 

        self.peak_resolutions = np.loadtxt(os.path.join(test_path,'label.txt'), dtype=str)   
        
        
    def __getitem__(self, idx):
        
        ############ for test  ####################
        data_name = self.test_graphs[idx]

        test_G = load_network(self.test_graphs[idx])
        test_torch_G = from_networkx(test_G).to(self.args.device)
        test_torch_G.x = torch.ones((test_torch_G.num_nodes, 1), dtype=torch.float).to(self.args.device)

        test_community_label = self.test_community_labels[idx]
        _, nmi_score, peak_resolution = self.peak_resolutions[idx]

        nmi_score = torch.tensor([float(nmi_score)]).to(self.args.device)
        peak_resolution = torch.tensor([float(peak_resolution)]).to(self.args.device)

        return data_name, test_G, test_torch_G, test_community_label, peak_resolution
        
        
    def __len__(self):
        return len(self.test_graphs)


def inference(args, model_path):
    # Load the model
    if args.ablation == 'full_model':
        print("Full model")
        model = mainnet_ResolNet(args).to(args.device)

    elif args.ablation == 'only_MLP':
        print("Only MLP")
        model = onlyMLP_ResolNet(args).to(args.device)
    
    elif args.ablation == 'only_GCN':
        print("Only GCN")
        model = onlyGCN_ResolNet(args).to(args.device)

    model = torch.load(model_path)
    # model.load_state_dict(model_path1)
    model.eval()

    # Prepare the test dataset
    test_dataset = test_graph_loader(args, args.test_path)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # Perform inference
    results = []

    with torch.no_grad():
        for data_name, test_G, test_torch_G, test_community_label, peak_resolution in test_dataset:
            out = model(test_G, test_torch_G)

            # Scale
            min_value = nn.Parameter(torch.tensor([0.1])).to(args.device)
            range_value = nn.Parameter(torch.tensor([80.0])).to(args.device)
            out = min_value + range_value * out

            out_new = out.item()
            predicted_community = detect_communities_leiden(test_G, resolution=0.1)#out_new

            true_communities = load_ground_truth(test_community_label)
            #original_nmi_score = calculate_nmi(true_communities, predicted_community)
            #original_nmi_score_torch = torch.tensor([float(original_nmi_score)]).to(args.device)
            #nmi_loss = criterion(nmi_score, original_nmi_score_torch)

            nmi_score = calculate_nmi(true_communities, predicted_community)

            results.append((data_name, nmi_score, out_new))
            print(f'NMI score for {data_name}: {nmi_score:.8f} {1:.8f}')#
            print(f'{test_community_label}')
            print('')

    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT', help='path of test dataset') # path/to/yours
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model_path', type=str, default='./ckpt/TC1-best.pt', help='path to the best model')
    parser.add_argument('--output_file', type=str, default='vis/inference_result.txt', help='path to save the output results')
    parser.add_argument('--testset', type=str, default='real', help='-')
    parser.add_argument('--ablation', type=str, default='full_model', help='-')

    # Dimension for first MLP (sub)
    parser.add_argument('--sMLP_idim', type=int, default=7, help='dimension of input tensor') 
    parser.add_argument('--sMLP_hdim1', type=int, default=256, help='dimension of 1st MLP hidden layer') 
    parser.add_argument('--sMLP_hdim2', type=int, default=32, help='dimension of 2nd MLP hidden layer') 
    parser.add_argument('--sMLP_odim', type=int, default=10, help='dimension of output tensor') 

    # Dimension for GCN 
    # input dimension fixed as 1
    parser.add_argument('--GCN_hdim1', type=int, default=64, help='dimension of 1st GCN hidden layer') 
    parser.add_argument('--GCN_hdim2', type=int, default=64, help='dimension of 2nd GCN hidden layer') 
    parser.add_argument('--GCN_odim', type=int, default=10, help='dimension of output tensor') 

    # Dimension for final MLP
    parser.add_argument('--fMLP_idim', type=int, default=20, help='dimension of input tensor') 
    parser.add_argument('--fMLP_hdim1', type=int, default=256, help='dimension of 1st MLP hidden layer') 
    parser.add_argument('--fMLP_hdim2', type=int, default=32, help='dimension of 2nd MLP hidden layer') 
    parser.add_argument('--fMLP_odim', type=int, default=1, help='dimension of output tensor') 

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = inference(args, args.model_path)
    #save_results_to_txt(results, args.output_file)
    print(f'Results saved to {args.output_file}')
