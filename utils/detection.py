import networkx as nx
import igraph as ig # TODO=it can be replaced with network x?
import leidenalg
import community as community_louvain

def detect_communities_leiden(G, resolution=1.0):
    node_id_map = {node: idx for idx, node in enumerate(G.nodes())}
    g = ig.Graph(edges=[(node_id_map[u], node_id_map[v]) for u, v in G.edges()], directed=False)
    g.vs["name"] = list(G.nodes())
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    communities = [list(g.vs[part]['name']) for part in partition]
    return communities

def detect_communities_louvain(G, resolution=1.0):
    partition = community_louvain.best_partition(G, resolution=resolution)
    community_to_nodes = {}
    for node, community in partition.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())