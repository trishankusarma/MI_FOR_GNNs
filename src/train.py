import torch
from torch_geometric.loader import DataLoader

from src.config import HyperParameters
from src.data_loader import corrupt_input_data
from src.dataset_generation import AlgorithmicReasoningDataset
from src.model import MinAggGNNLayer
from src.scores import first_order_taylor_weight_grad
from src.circuit_discovery import discover_circuit, evaluate_circuit_sufficiency

ROOT_DATA_PATH = "./data"

def run_epoch(model, loader, optimizer):
    """
    Runs a single epoch iterating through mini-batches provided by the PyG DataLoader.
    """
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, batch_data in enumerate(loader):
        # 1. Clear gradients for the new batch
        optimizer.zero_grad()
        
        # 2. Forward Pass on the batched graph matrix
        output = model(batch_data.x, batch_data.edge_index, batch_data.edge_weight)
        
        # Calculate MSE loss matching the current batch size targets
        loss = torch.mean((output[:, 0] - batch_data.y) ** 2) 
        
        # 3. Backward Pass
        loss.backward()
        
        # 4. Step the optimizer to actually update the weights!
        optimizer.step()
        
        # Track running loss safely scales by number of graphs in the current batch
        epoch_loss += loss.item() * batch_data.num_graphs

    # Calculate average loss over the entire dataset footprint
    avg_loss = epoch_loss / len(loader.dataset)
    
    # 5. Score Tracking: Print WeightGrad sensitivity once per epoch closure
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'mlp.weight' in name:
                scores = first_order_taylor_weight_grad(param)
                print(f"Epoch Loss: {avg_loss:.4f} | Layer {name} Current Scores:\n", scores)
                
    return scores

if __name__=="__main__":
    # step 1: Load configurations and Dataset
    config = HyperParameters()
    
    # Instantiate your 10k production graph dataset
    dataset = AlgorithmicReasoningDataset(root=ROOT_DATA_PATH)
    
    # Wrap dataset in PyG's DataLoader (Add batch_size to your config file if missing!)
    batch_size = getattr(config, 'batch_size', 32) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # step 2: Initialize model architecture
    model = MinAggGNNLayer(config.in_channels, config.out_channels)
    
    # step 3: Initialize tracking optimizers
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    # step 4: Run training loop across batches
    print(f"Starting training loop across {len(loader)} mini-batches...")
    for epoch in range(config.num_epochs):
        print(f"--- Epoch {epoch+1}/{config.num_epochs} ---")
        run_epoch(model=model, loader=loader, optimizer=optimizer)
    
    # CIRCUIT DISCOVERY PHASE (Post-Training)
    print("\n--- Initiating Mechanistic Evaluation ---")
    
    # Grab a clean, single batch of graphs out of our loader to evaluate the circuit
    eval_batch = next(iter(loader))
    
    # 1. Get corrupted evaluation baseline (your logic functions flawlessly on batched data!)
    corrupted_batch = corrupt_input_data(eval_batch)
    
    # 2. Discover Circuit Mask using the target evaluation batch
    mask = discover_circuit(model, eval_batch, corrupted_batch, K=config.k)
    
    # 3. Test if the isolated sub-circuit parameters hold up on the batched sample
    evaluate_circuit_sufficiency(model, eval_batch, mask)