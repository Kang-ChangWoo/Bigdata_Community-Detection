import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import torch.optim as optim
import argparse
import igraph as ig
import leidenalg
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
from node2vec import Node2Vec



#################### Utilty Functions #############################

def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

def load_ground_truth(file_path):
    node_to_community = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node, community = int(parts[0]), int(parts[1])
                node_to_community[node] = community

    community_to_nodes = {}
    for node, community in node_to_community.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())

# Calculating NMI Score
def calculate_nmi(true_communities, detected_communities):
    true_labels = {}
    for i, community in enumerate(true_communities):
        for node in community:
            true_labels[node] = i
    detected_labels = {}
    for i, community in enumerate(detected_communities):
        for node in community:
            detected_labels[node] = i

    nodes = sorted(set(true_labels) | set(detected_labels))
    true_labels_vector = [true_labels[node] for node in nodes]
    detected_labels_vector = [detected_labels.get(node, -1) for node in nodes]

    return normalized_mutual_info_score(true_labels_vector, detected_labels_vector)

def detect_communities_leiden(G, resolution=1.0):
    node_id_map = {node: idx for idx, node in enumerate(G.nodes())}
    g = ig.Graph(edges=[(node_id_map[u], node_id_map[v]) for u, v in G.edges()], directed=False)
    g.vs["name"] = list(G.nodes())
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    communities = [list(g.vs[part]['name']) for part in partition]

    return communities

def eval(true_communities, detected_communities):
    nmi_score = calculate_nmi(true_communities, detected_communities)
    return nmi_score


#################### Training Functions #############################

class Embedding(G):
    def __init__(self):
        super(Embedding, self).__init__()
    
    def embedding_metainfo(self, G):
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = sum(dict(G.degree()).values()) / num_nodes
        density = nx.density(G)
        avg_clustering_coefficient = nx.average_clustering(G)
        transitivity = nx.transitivity(G)
        
        metainfo = torch.tensor([
            num_nodes,
            num_edges,
            avg_degree,
            density,
            avg_clustering_coefficient,
            transitivity
        ], dtype=torch.float32)
        
        return metainfo
    
    def embedding_graph(self, G):
        node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        nodes = list(G.nodes)
        embeddings = torch.tensor([model.wv[str(node)] for node in nodes], dtype=torch.float32)

        return embeddings
    
    def forward(self,G):
        
        meta = self.embedding_metainfo(G)
        graph = self.embedding_graph(G)
        meta_expanded = meta.unsqueeze(0).repeat(graph.size(0), 1)  # Shape: (N, 6)
        embedding_total = torch.cat((graph, meta_expanded), dim=1)  # Shape: (N, D+6)

        return embedding_total


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.first = nn.Sequential(
                        nn.Linear(in_dim, 256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(32),
                nn.Linear(32, 1)
        )


# txt 파일 형태로 불러온다고 가정함 (1-1.dat 이 10개정도?)
class OursData():
    pass


def train(model, optimizer, train_loader, epoch, criterion):

    for i, (txt_names) in enumerate(train_loader):
       
        # training data 및 gt 데이터 불러오고 networkx 라이브러리 그래프 형태로 불러오기
        # 불러온 그래프를 임베딩하기
        
        txt_name = txt_names[i]
        
        train_communities = load_network(txt_name)
        gt_communities = load_ground_truth(txt_name)

        embedding_total = Embedding(train_communities)
        total_result = model(embedding_total)
        
        
        
        
        
        # pseudo-GT 의 최적 resolution 구하기
        r=0
        resolutions, scores = [], []
        for _ in tqdm(range(50)):
            r += 1
            detected_communities = detect_communities_leiden(train_communities, resolution=r)
            nmi_score = eval(gt_communities, train_communities)
            resolutions.append(r)
            scores.append(nmi_score)
        
        best_resolution = resolutions[scores.index(max(scores))]
        
        # Regressor를 통해 구한 resolution과 pseudo-GT 의 최적 resolution과 MSE LOSS 비교
        loss = criterion(total_result, torch.tensor([best_resolution], dtype=torch.float32))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
        

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = OursData()
    train_loader = torch.utils.data.DataLodaer()

    model = Regressor()
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    criterion = nn.MSELoss()

    for epoch in range(args.epoch):
        model.train()
        
        loss = train(model, optimizer, train_loader, epoch, criterion)
        print(f'Epoch [{epoch+1}/{args.epoch}], Loss: {loss:.4f}')

if __name__ == "__main__":
    main()




