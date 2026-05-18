import torch

from src.config import HyperParameters
from src.data_loader import CreateDataLoader, corrupt_input_data
from src.model import MinAggGNNLayer
from src.scores import first_order_taylor_weight_grad
from src.circuit_discovery import discover_circuit, evaluate_circuit_sufficiency

def run_epoch(model, batch_data):
    model.zero_grad()
    
    # 1. Clean Forward Pass
    output = model(batch_data.x, batch_data.edge_index, batch_data.edge_weight)
    
    # Simple MSE Loss for our toy example
    loss = torch.mean((output[:, 0] - batch_data.y) ** 2) 
    
    # 2. Backward Pass to populate gradients
    loss.backward()
    
    # 3. Calculate absolute weight gradient interaction (Baseline WeightGrad)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'mlp.weight' in name:
                scores = first_order_taylor_weight_grad(param)
                print(f"Layer {name} Scores:\n", scores)
    return scores

if __name__=="__main__":
    # step 1: load data
    config = HyperParameters()
    data = CreateDataLoader()
    # step 2: initialize model
    model = MinAggGNNLayer(config.in_channels, config.out_channels)
    # step 3: initialize optimizers
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    # step 4: run epochs
    for epoch in range(config.num_epochs):
        run_epoch(model=model, batch_data=data)
    
    # 1. Get corrupted baseline
    corrupted_data = corrupt_input_data(data)
    
    # 2. Run Circuit Discovery for K=2 (since your matrix only has 2 dominant values!)
    mask = discover_circuit(model, data, corrupted_data, K=config.k)
    
    # 3. Test if those 2 parameters are sufficient to hold the algorithm together
    evaluate_circuit_sufficiency(model, data, mask)