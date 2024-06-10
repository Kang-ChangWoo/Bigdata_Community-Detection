
import os, argparse, random, glob

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader #, Dataset, 

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader as g_DataLoader

import networkx as nx
import igraph as ig # TODO=it can be replaced with network x?
import leidenalg

from sklearn.preprocessing import StandardScaler


"""
1. Helper function.
"""
def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


def extract_meta_info(G, args):
    meta_info = {}

    meta_info['n_nodes'] = G.number_of_nodes()
    meta_info['n_edges'] = G.number_of_edges()
    meta_info['avg_deg'] = sum(dict(G.degree()).values()) / meta_info['n_nodes']
    meta_info['density'] = nx.density(G)
    meta_info['n_connected_comp'] = nx.number_connected_components(G)
    meta_info['avg_cluster'] = nx.average_clustering(G)
    meta_info['transitivity'] = nx.transitivity(G)
    # meta_info['avg_short_length'] = nx.average_shortest_path_length(G) # it takes too much time!
    # meta_info['diameter'] = nx.diameter(G) # it takes too much time!

    # [ETA for parameters from `TC1/TC1-1/1-1.dat`]
    #   all parameters: 6m 30s takes
    #   number_of_nodes: 0.0s takes
    #   number_of_edges: 0.0s takes
    #   average_degree: 0.0s takes
    #   density: 0.0s takes
    #   num_connected_components: 3m 16.5s takes
    #   diameter: 3m 13.1s takes
    #   average_clustering: 1.2s takes

    # [Actual results from `TC1/TC1-1/1-1.dat`]
    #   Nodes: 10000
    #   Edges: 83441
    #   Average Degree: 16.6882
    #   Density: 0.0016
    #   Number of Connected Components: 1
    #   Average Shortest Path Length: 3.6290
    #   Diameter: 7
    #   Average Clustering Coefficient: 0.4526
    
    meta_vec = torch.tensor([[meta_info['n_nodes'],
                            meta_info['n_edges'],
                            meta_info['avg_deg'],
                            meta_info['density'],
                            meta_info['n_connected_comp'],
                            meta_info['avg_cluster'],
                            meta_info['transitivity'],]],
                            dtype=torch.float32) #problem?

    scaler = StandardScaler()
    scaled_vec = scaler.fit_transform(meta_vec)
    return torch.from_numpy(scaled_vec).to(args.device).float()


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



"""
3. Neural network tool-kit.
   (Dataloader, Network model)
"""
class graph_loader(torch.utils.data.Dataset):
    def __init__(self, args, file_path):
        self.file_path = file_path
        
        self.graphs = glob.glob(os.path.join(file_path, '*-*.dat' ))
        self.graphs = sorted(self.graphs)

        self.community_labels = glob.glob(os.path.join(file_path, '*-*-c.dat' ))
        self.community_labels = sorted(self.community_labels)

        self.args = args

        self.peak_resolutions = np.loadtxt(os.path.join(file_path,'label.txt'), dtype=str)

    def __getitem__(self, idx):
        G = load_network(self.graphs[idx])
        torch_G = from_networkx(G).to(self.args.device)
        torch_G.x = torch.ones((torch_G.num_nodes, 1), dtype=torch.float).to(self.args.device)

        #community_label = self.community_labels[idx]
        community_label = 'testing'
        _, rmi_score, peak_resolution = self.peak_resolutions[idx]

        peak_resolution = torch.tensor([float(peak_resolution)]).to(self.args.device)
        return G, torch_G, community_label, peak_resolution
        
    def __len__(self):
        return len(self.graphs)



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
        meta_info = extract_meta_info(G, self.args)
        out1 = self.sub_MLP(meta_info) #10d vector
        out1 = out1.squeeze()

        #graph_info = torch_G
        x = torch_G.x.to(self.args.device)
        edge_index = torch_G.edge_index.to(self.args.device)
        #batch = torch_G.batch.to(self.args.device)
        out2 = self.sub_GCN(x, edge_index) #10d vector , batch

        out3 = self.fin_MLP(torch.cat((out1, out2), dim=0))
        return out3

# if __name__ == "__main__":
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('Current Device:', device)
    
#     parser = argparse.ArgumentParser(description='Detect communities in a network.')
#     parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./1-1.dat")
#     args = parser.parse_args()
    
#     G = load_network(args.networkFile)
#     data = from_networkx(G)

#     # print(G)      // Graph with 10000 nodes and 83441 edges
#     # print(data)   //Data(edge_index=[2, 166882], num_nodes=10000)

#     # 노드 간의 연결 관계만 가지고 있음 , 구체적인 특성 정보는 없기 때문에 1로 설정하였스무니다
#     data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
#     data = data.to(device)

#     model = GCN(hidden_dim=16, output_dim=2).to(device)

#     model.eval()
#     out = model(data.x, data.edge_index)
#     print(out.shape) 





def validate(net, val_loader):
    # =======    
    # Call the loss criterion and loader
    # =======
    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    data_iter = len(val_loader)

    avg_loss = 0.0

    with torch.no_grad():
        for i, (G, torch_G, community_label, peak_resolution) in enumerate(val_loader, 0):

            out = net(G, torch_G)

            # Scale
            min_value = nn.Parameter(torch.tensor([0.1])).to(args.device)
            range_value = nn.Parameter(torch.tensor([80.0])).to(args.device)
            out = min_value + range_value * out
    
            res_loss = criterion(out, peak_resolution)
            loss = res_loss

            avg_loss += loss.item() / data_iter

        print(f'Test done. (Average loss: {avg_loss:.8f})')

    return avg_loss




"""
4. Model train
"""

def train(args):
    # seed fixed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    dataset = graph_loader(args, args.data_path)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = train_dataset
    test_loader = test_dataset

    #train_loader = DataLoader(train_dataset, shuffle=False, drop_last=True, pin_memory=True )
    #test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, pin_memory=True )

    #train_loader = g_DataLoader(train_loader, batch_size=10, shuffle=True)

    model = mainnet_ResolNet(args).to(args.device)
    #model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    criterion = nn.MSELoss()
    print("lets go!")

    for epoch in range(args.epoch):
        model.train()
        r = 1.0

        # TODO= add up tqdm
        for i, (G, torch_G, community_label, peak_resolution) in tqdm(enumerate(train_loader, 0), desc=f'{epoch}-th learning', total=len(train_loader)):
            optimizer.zero_grad()

            out = model(G, torch_G)

            # Scale
            min_value = nn.Parameter(torch.tensor([0.1])).to(args.device)
            range_value = nn.Parameter(torch.tensor([80.0])).to(args.device)
            out = min_value + range_value * out
    
            res_loss = criterion(out, peak_resolution)
            loss = res_loss
            
            # pred_communities = detect_communities_leiden(out)

            # resolutions, scores = [], []
            # for _ in tqdm(range(50)):
            #     r += 1
            #     detected_communities = detect_communities_leiden(train_communities, resolution=r)
            #     nmi_score = eval(gt_communities, train_communities)
            #     resolutions.append(r)
            #     scores.append(nmi_score)
            
            # best_resolution = resolutions[scores.index(max(scores))]
            
            # Regressor를 통해 구한 resolution과 pseudo-GT 의 최적 resolution과 MSE LOSS 비교
            # loss = criterion(total_result, torch.tensor([best_resolution], dtype=torch.float32))
            
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{args.epoch}], Loss: {loss.item():.4f}')
            # 어떻게 비교하지??
            # Loss 어떻게 구성할지 논의 필요
        
        model.eval()
        eval_performance = validate(model, test_loader)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', type=int, default=150, help='epoch')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--w_peak', type=float, default=1.0, help='weight for peak loss')
    parser.add_argument('--w_comm', type=float, default=1.0, help='weight for community loss')
    parser.add_argument('--data_path', type=str, default='path/to/yours', help='path of dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    # Dimension for first MLP (sub)
    parser.add_argument('--sMLP_idim', type=int, default=7, help='dimension of input tensor') # WHAT?
    parser.add_argument('--sMLP_hdim1', type=int, default=256, help='dimension of 1st MLP hidden layer') # WHAT?
    parser.add_argument('--sMLP_hdim2', type=int, default=32, help='dimension of 2nd MLP hidden layer') # WHAT?
    parser.add_argument('--sMLP_odim', type=int, default=10, help='dimension of output tensor') # WHAT?

    # Dimension for GCN 
    # input dimension fixed as 1
    parser.add_argument('--GCN_hdim1', type=int, default=64, help='dimension of 1st GCN hidden layer') # WHAT?
    parser.add_argument('--GCN_hdim2', type=int, default=64, help='dimension of 2nd GCN hidden layer') # WHAT?
    parser.add_argument('--GCN_odim', type=int, default=10, help='dimension of output tensor') # WHAT?

    # Dimension for final MLP
    parser.add_argument('--fMLP_idim', type=int, default=20, help='dimension of input tensor') # WHAT?
    parser.add_argument('--fMLP_hdim1', type=int, default=256, help='dimension of 1st MLP hidden layer') # WHAT?
    parser.add_argument('--fMLP_hdim2', type=int, default=32, help='dimension of 2nd MLP hidden layer') # WHAT?
    parser.add_argument('--fMLP_odim', type=int, default=1, help='dimension of output tensor') # WHAT?

    # Dimension for final MLP
    
    args = parser.parse_args()

    train(args)