import torch
import torch_geometric.data as data

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