import random
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from src.config import HyperParameters

ROOT_DATA_PATH = "./data"
MAX_DISTANCE_THRESHOLD = 1e5 

class AlgorithmicReasoningDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=500, min_nodes=50, max_nodes=200, edge_prob=0.5, transform=None, pre_transform=None):
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.edge_prob = edge_prob
        super(AlgorithmicReasoningDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['algorithmic_data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        for _ in tqdm(range(self.num_graphs), desc="Processing Graphs"):
            # 1. Generate a random graph topology
            num_nodes = random.randint(self.min_nodes, self.max_nodes)
            G = nx.erdos_renyi_graph(n=num_nodes, p=self.edge_prob, directed=True)
            
            if len(G.edges) == 0:
                continue # Skip empty graphs
                
            # Assign random integer weights to edges
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = float(random.randint(1, 10))

            # Pick a random source node
            source_node = random.randint(0, num_nodes - 1)

            # 2. Calculate Ground Truth using classical algorithms
            distances = nx.single_source_dijkstra_path_length(G, source_node, weight='weight')
            
            # 3. Construct Tensors
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor([G.edges[u, v]['weight'] for u, v in G.edges], dtype=torch.float)

            # Parallel feature tracks
            x_sp = torch.full((num_nodes, 1), MAX_DISTANCE_THRESHOLD, dtype=torch.float)
            x_sp[source_node] = 0.0

            x_bfs = torch.zeros((num_nodes, 1), dtype=torch.float)
            x_bfs[source_node] = 1.0

            # Concatenate inputs to form parallel lanes: Shape [num_nodes, 2]
            x = torch.cat([x_sp, x_bfs], dim=-1)

            # Target labels: Map networkx results back to sequential tensor
            y_dist = torch.zeros((num_nodes, 1), dtype=torch.float)
            for node in range(num_nodes):
                y_dist[node] = distances.get(node, MAX_DISTANCE_THRESHOLD)

            # Pack everything into a PyG Data wrapper
            graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y_dist)
            data_list.append(graph_data)

        # FIXED: Removed the unused standalone collate call; self.save handles it completely
        self.save(data_list, self.processed_paths[0])

if __name__ == "__main__":
    config = HyperParameters()
    print("Generating training dataset...")
    dataset = AlgorithmicReasoningDataset(
        root=ROOT_DATA_PATH, 
        num_graphs=config.num_graphs,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes, 
        edge_prob=config.edge_prob, 
        transform=config.transform, 
        pre_transform=config.pre_transform
    )
    print(f"Successfully generated and loaded {len(dataset)} graphs.")