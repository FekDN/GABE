# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

import torch
from transformers import GPT2Model
from typing import List, Dict
import numpy as np

def eigen_basis_extractor(weights_list: List[torch.Tensor]):
    """
    Extracts an Operator-Centric representation from a list of weight tensors.
    """
    stacked = torch.stack(weights_list)
    w_bar = torch.mean(stacked, dim=0)
    
    L, d1, d2 = stacked.shape
    flattened = stacked.view(L, -1)
    mean_flat = w_bar.view(-1)
    centered = flattened - mean_flat
    
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    B_list = []
    for k in range(L):
        B_list.append((S[k] * Vh[k]).view(d1, d2))
    
    B_stacked = torch.stack(B_list)
    coeffs = U
    
    return w_bar, B_stacked, coeffs

def format_bytes(size: int) -> str:
    """Formats bytes into readable format (KB, MB)."""
    power = 1024
    n = 0
    power_labels = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size >= power and n < len(power_labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def analyze_layer_group(layer_group_name: str, weights_list: List[torch.Tensor]):
    """
    Performs a full analysis for one group of layers and outputs a report.
    """
    print(f"\n{'='*25} Analysis of a group of layers: {layer_group_name} {'='*25}")
    
    num_layers, in_features, out_features = len(weights_list), weights_list[0].shape[0], weights_list[0].shape[1]
    print(f"Extracted {num_layers} shape tensors ({in_features}, {out_features}).")

    # 1. Extracting the Operator-Centric Representation
    w_bar, B, coeffs = eigen_basis_extractor(weights_list)
    
    # 2. Dimensional and precision analysis
    original_size_bytes = num_layers * in_features * out_features * 4
    print(f"Original size ({num_layers} tensors): {format_bytes(original_size_bytes)}")
    
    # --- Scenario A: Accurate reconstruction (for verification) ---
    K_exact = num_layers
    W_original = weights_list[0]
    W_recon_exact = w_bar.clone()
    for k in range(K_exact):
        W_recon_exact += coeffs[0, k] * B[k]
    is_correct = torch.allclose(W_original, W_recon_exact, atol=1e-4)
    print(f"\n--- Verification (K={K_exact}):")
    print(f"  -> The reconstruction is accurate: {is_correct} (MSE: {torch.mean((W_original-W_recon_exact)**2):.2e})")

    # --- Scenario B: Low-Rank Approximation (for Compression) ---
    # Let's try several ranks for a more complete analysis.
    ranks_to_test = [1, 2, 4, 8]
    
    print("\n--- Compression and error analysis for different ranks (K) ---")
    print("-" * 60)
    print(f"{'Ранг (K)':>10} | {'Compression ratio':>20} | {'MSE errors':>15}")
    print("-" * 60)
    
    for K_approx in ranks_to_test:
        if K_approx >= num_layers: continue
        
        # Size calculation
        mean_size_bytes = w_bar.numel() * 4
        basis_size_bytes_approx = B[:K_approx].numel() * 4
        coeffs_size_bytes_approx = coeffs[:, :K_approx].numel() * 4
        compressed_size_bytes_approx = mean_size_bytes + basis_size_bytes_approx + coeffs_size_bytes_approx
        compression_ratio = original_size_bytes / compressed_size_bytes_approx
        
        # Error calculation
        W_recon_approx = w_bar.clone()
        for k in range(K_approx):
            W_recon_approx += coeffs[0, k] * B[k]
        mse = torch.mean((W_original - W_recon_approx)**2).item()
        
        print(f"{K_approx:>10} | {compression_ratio:>19.2f}x | {mse:>14.2e}")
    print("-" * 60)

# ============================================================
# Main block
# ============================================================

if __name__ == "__main__":
    print("Loading the GPT-2 model...")
    model = GPT2Model.from_pretrained('gpt2')
    
    # Collect weights into dictionaries by groups
    layer_groups = {
        "FFN1 (c_fc)": [layer.mlp.c_fc.weight.T for layer in model.h],
        "FFN2 (c_proj)": [layer.mlp.c_proj.weight.T for layer in model.h],
        "Attention (c_attn)": [layer.attn.c_attn.weight.T for layer in model.h],
        "Attention (c_proj)": [layer.attn.c_proj.weight.T for layer in model.h],
    }
    
    # Run the analysis for each group
    for name, weights in layer_groups.items():
        analyze_layer_group(name, weights)
