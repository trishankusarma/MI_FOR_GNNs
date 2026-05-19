import torch
from torch_geometric.loader import DataLoader

from src.config import HyperParameters
from src.data_loader import corrupt_input_data
from src.dataset_generation import AlgorithmicReasoningDataset
from src.model import MinAggGNNLayer
from src.scores import first_order_taylor_weight_grad
from src.circuit_discovery import discover_circuit, evaluate_circuit_sufficiency

ROOT_DATA_PATH = "./data"
MAX_DISTANCE_THRESHOLD = 1e5 

def run_epoch(model, loader, optimizer):
    """
    Runs a single epoch iterating through mini-batches provided by the PyG DataLoader.
    """
    model.train()
    epoch_loss = 0.0
    nodes_counted = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 1. Clear gradients for the new batch
        optimizer.zero_grad()

        # 2. Forward Pass on the batched graph matrix
        # Both tensors are now cleanly shaped [Num_Nodes, 1]
        output = model(batch_data.x, batch_data.edge_index, batch_data.edge_weight)

        # 3. Masked Loss (Only compute MSE for reachable nodes where y < 1e5)
        reachable_mask = (batch_data.y < MAX_DISTANCE_THRESHOLD).squeeze(-1)  # Returns a boolean tensor

        if reachable_mask.sum() == 0:
            continue # Skip batch if no reachable target nodes exist
        
        # 4. Compute MSE only on the masked elements
        loss = torch.mean((output.squeeze(-1)[reachable_mask] - batch_data.y.squeeze(-1)[reachable_mask]) ** 2)
        
        # 5. Backward Pass
        loss.backward()

        # 5. Gradient Clipping
        # Prevents any residual mathematical anomalies from shattering the weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 6. Step the optimizer to actually update the weights!
        optimizer.step()
        
        # Track loss relative to the number of valid nodes evaluated
        epoch_loss += loss.item() * reachable_mask.sum().item()
        nodes_counted += reachable_mask.sum().item()

    # Calculate average loss over the entire dataset footprint
    avg_loss = epoch_loss / nodes_counted if nodes_counted > 0 else 0
    print(f"Epoch Loss: {avg_loss:.6f}")

if __name__=="__main__":
    # step 1: Load configurations and Dataset
    config = HyperParameters()
    
    # Instantiate your 10k production graph dataset
    dataset = AlgorithmicReasoningDataset(root=ROOT_DATA_PATH)
    
    # Wrap dataset in PyG's DataLoader (Add batch_size to your config file if missing!)
    batch_size = getattr(config, 'batch_size', 32) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # step 2: Initialize model architecture
    model = MinAggGNNLayer(config.in_channels, config.hidden_layers, config.out_channels)
    
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