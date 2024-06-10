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
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.utils import from_networkx

from precompute_peak import *
from torch.utils.data import DataLoader  





"""
2. Community detection algorithm.
"""
def detect_communities_leiden(G, resolution=1.0):
    node_id_map = {node: idx for idx, node in enumerate(G.nodes())}
    g = ig.Graph(edges=[(node_id_map[u], node_id_map[v]) for u, v in G.edges()], directed=False)
    g.vs["name"] = list(G.nodes())
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    communities = [list(g.vs[part]['name']) for part in partition]
    return communities

def detect_communities_louvain(G, resolution_ = 1.0):
    partition = community_louvain.best_partition(G, resolution=resolution_)
    community_to_nodes = {}
    for node, community in partition.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())


def load_network(train_path):
    G = nx.Graph()
    with open(train_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

def load_community_labels(file_path):
    with open(file_path, 'r') as f:
        communities = [int(line.strip()) for line in f]
    return communities



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

        ############ for test  ####################
        # 저장구조 형태 변경 필요
        
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

class subnet_MLP(nn.Module):
    def __init__(self, args):
        super(subnet_MLP, self).__init__()

        self.linear1 = nn.Linear(args.sMLP_idim, args.sMLP_hdim1)
        self.linear2 = nn.Linear(args.sMLP_hdim1, args.sMLP_hdim2)
        self.output = nn.Linear(args.sMLP_hdim2, args.sMLP_odim)
        
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.batch_norm1 = nn.BatchNorm1d(args.sMLP_hdim1)
        self.batch_norm2 = nn.BatchNorm1d(args.sMLP_hdim2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        #x = self.batch_norm1(x)

        x = self.linear2(x)
        x = self.leaky_relu(x)
        #x = self.batch_norm2(x)

        x = self.output(x)
        return x 

class subnet_GCN(nn.Module):
    def __init__(self, args):
        super(subnet_GCN, self).__init__()
        self.conv1 = GCNConv(1, args.GCN_hdim1)  
        self.conv2 = GCNConv(args.GCN_hdim1, args.GCN_hdim2)
        self.conv3 = GCNConv(args.GCN_hdim2, args.GCN_odim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    # , batch
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv3(x, edge_index)
        #x = self.leaky_relu(x)

        #x = global_mean_pool(x, 1) # , batch

        x = torch.mean(x, dim=0)

        return x
        
class finnet_MLP(nn.Module):
    def __init__(self, args):
        super(finnet_MLP, self).__init__()

        self.linear1 = nn.Linear(args.fMLP_idim, args.fMLP_hdim1)
        self.linear2 = nn.Linear(args.fMLP_hdim1, args.fMLP_hdim2)
        self.output = nn.Linear(args.fMLP_hdim2, args.fMLP_odim)
        
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.batch_norm1 = nn.BatchNorm1d(args.fMLP_hdim1)
        self.batch_norm2 = nn.BatchNorm1d(args.fMLP_hdim2)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        #x = self.batch_norm1(x)

        x = self.linear2(x)
        x = self.leaky_relu(x)
        #x = self.batch_norm2(x)

        x = self.output(x)
        x = self.sigmoid(x)
        return x 


class mainnet_ResolNet(nn.Module):
    def __init__(self, args):
        super(mainnet_ResolNet, self).__init__()
        self.args = args

        self.sub_MLP = subnet_MLP(self.args).to(args.device) 
        self.sub_GCN = subnet_GCN(self.args).to(args.device)

        self.fin_MLP = finnet_MLP(self.args).to(args.device)

    def forward(self, G, torch_G):
        meta_info = extract_meta_info(G, self.device)
        out1 = self.sub_MLP(meta_info) #10d vector
        out1 = out1.squeeze()

        #graph_info = torch_G
        x = torch_G.x.to(self.args.device)
        edge_index = torch_G.edge_index.to(self.args.device)
        #batch = torch_G.batch.to(self.args.device)
        out2 = self.sub_GCN(x, edge_index) #10d vector , batch

        out3 = self.fin_MLP(torch.cat((out1, out2), dim=0))
        return out3



def inference(args, model_path):
    # Load the model
    model = mainnet_ResolNet(args).to(args.device)
    model = torch.load(model_path)
    #model.load_state_dict(torch.load(model_path))
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
            predicted_community = detect_communities_louvain(test_G, resolution_=out_new)

            true_communities = load_ground_truth(test_community_label)
            #original_nmi_score = calculate_nmi(true_communities, predicted_community)
            #original_nmi_score_torch = torch.tensor([float(original_nmi_score)]).to(args.device)
            #nmi_loss = criterion(nmi_score, original_nmi_score_torch)

            nmi_score = calculate_nmi(true_communities, predicted_community)

            results.append((data_name, nmi_score, out_new))
            print(f'NMI score for {data_name}: {nmi_score:.8f} {out_new:.8f}')

    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--test_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT', help='path of test dataset') # path/to/yours
    #parser.add_argument('--test_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT', help='path of test dataset') # path/to/yours
    parser.add_argument('--test_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/TC1-all_including-GT', help='path of test dataset') # path/to/yours
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model_path', type=str, default='./correction-best.pt', help='path to the best model')
    parser.add_argument('--output_file', type=str, default='inference_result.txt', help='path to save the output results')
    parser.add_argument('--testset', type=str, default='TC1', help='-')
    #parser.add_argument('--pth', type=str, default='basic', help='-')

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
