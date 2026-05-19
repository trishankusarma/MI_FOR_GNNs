import torch

def discover_circuit(model, clean_data, corrupted_data, K=10):
    """
    Implements Algorithm 1: Drops elements until only the Top-K circuit remains.
    """
    model.eval()
    
    # 1. Run clean and corrupted forward passes to establish baselines
    with torch.no_grad():
        clean_out = model(clean_data.x, clean_data.edge_index, clean_data.edge_weight)
        corr_out = model(corrupted_data.x, corrupted_data.edge_index, corrupted_data.edge_weight)
    
    # 2. Fetch our structural scores (using the WeightGrad or EAP scoring logic)
    # For this scratch example, we'll extract the scores matrix you just printed
    model.zero_grad()
    clean_out_grad = model(clean_data.x, clean_data.edge_index, clean_data.edge_weight)
    loss = torch.mean((clean_out_grad[:, 0] - clean_data.y) ** 2)
    loss.backward()
    
    # Grab the target parameter matrix
    target_param = next(p for name, p in model.named_parameters() if 'mlp' in name and 'weight' in name)
    
    # Let's use the first-order Taylor score for this demonstration
    scores = torch.abs(target_param * target_param.grad)
    
    # 3. Flatten and sort the scores descending
    flat_scores = scores.flatten()
    sorted_values, sorted_indices = torch.sort(flat_scores, descending=True)
    
    # 4. Select the top K elements for our circuit mask
    circuit_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    top_k_indices = sorted_indices[:K]
    circuit_mask[top_k_indices] = True
    circuit_mask = circuit_mask.view_as(scores)
    
    print(f"--- Circuit Discovery Complete (K={K}) ---")
    print(f"Identified {K} critical parameters out of {flat_scores.numel()}.")
    
    return circuit_mask

def evaluate_circuit_sufficiency(model, clean_data, circuit_mask):
    """
    Tests FID-: Evaluates the model using ONLY the discovered circuit weights.
    """
    # Clone original weights so we don't permanently ruin our trained model
    target_param = next(p for name, p in model.named_parameters() if 'mlp' in name and 'weight' in name)
    original_weights = target_param.data.clone()
    
    try:
        # Zero out every parameter that is NOT in the circuit mask
        target_param.data = original_weights * circuit_mask.float()
        
        # Run inference with our stripped-down model
        with torch.no_grad():
            circuit_out = model(clean_data.x, clean_data.edge_index, clean_data.edge_weight)
            circuit_loss = torch.mean((circuit_out[:, 0] - clean_data.y) ** 2)
            
        print(f"Isolated Circuit Loss (FID-): {circuit_loss.item():.6f}")
        return circuit_loss.item()
        
    finally:
        # ALWAYS restore original weights after the experiment
        target_param.data = original_weights