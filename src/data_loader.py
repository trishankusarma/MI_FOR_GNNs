import torch
import torch_geometric.data as data

def CreateDataLoader():
    # A simple 4-node diamond graph: 0 -> 1 -> 3, 0 -> 2 -> 3
    edge_index = torch.tensor([[0, 0, 1, 2],
                               [1, 2, 3, 3]], dtype=torch.long) #<num_edges, 2>
    edge_weight = torch.tensor([1.0, 4.0, 2.0, 1.0], dtype=torch.float) # Path 0-1-3 = 3, Path 0-2-3 = 5 <num_edges, 1>
    
    num_nodes = 4
    source_node = 0
    
    # Track 1: Shortest Path (SP) initialization
    x_sp = torch.full((num_nodes, 1), 1e5) # Use a large number for infinity <num_nodes, 1>
    x_sp[source_node] = 0.0
    
    # Track 2: BFS (Reachability) initialization
    x_bfs = torch.zeros((num_nodes, 1)) # <num_nodes, 1>
    x_bfs[source_node] = 1.0
    
    # Concatenate parallel tracks: Shape <num_nodes, 2>
    x = torch.cat([x_sp, x_bfs], dim=-1)
    
    # True labels (Ground Truth distances from node 0)
    y_distances = torch.tensor([0.0, 1.0, 4.0, 3.0], dtype=torch.float) #<num_nodes, 1>
    
    return data.Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y_distances)

def corrupt_input_data(clean_data: data.Data) -> data.Data:
    """
    Corrupts data based on Eq 12:
    - Sets all edge weights to zero.
    - Flips the input feature logic (Source node becomes 'infinity', others become 0)
    """
    corrupted_data = clean_data.clone()
    
    # 1. Clear out the edge weights (make them zero or a dead neutral baseline)
    corrupted_data.edge_weight = torch.zeros_like(clean_data.edge_weight)
    
    # 2. Corrupt the node features x
    # Assuming x is shape [num_nodes, features] where index 0 is SP (Shortest Path)
    # We find where SP was 0 (the original source node) and make it infinity
    is_source = (clean_data.x[:, 0] == 0.0)
    
    corrupted_x = clean_data.x.clone()
    corrupted_x[is_source, 0] = 1e5       # Lobotomize the source node
    corrupted_x[~is_source, 0] = 0.0     # Zero out the rest
    
    corrupted_data.x = corrupted_x
    return corrupted_data