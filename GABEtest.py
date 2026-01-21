# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

import torch
from typing import List, Tuple, Dict, Any
import math
import warnings
import time
from collections import defaultdict
import os

from GABE import GABE

from transformers import GPT2Model, Conv1D
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def group_model_weights(model: nn.Module, layer_type: type) -> Dict[tuple, List[torch.Tensor]]:
    print(f"Searching for layers of type {layer_type.__name__}...")
    groups = defaultdict(list)
    for module in model.modules():
        if isinstance(module, layer_type):
            weight = module.weight.detach().clone()
            if isinstance(module, Conv1D):
                reshaped_weight = weight.T
            elif isinstance(module, nn.Conv2d):
                reshaped_weight = weight.view(weight.shape[0], -1)
            else:
                reshaped_weight = weight
            groups[reshaped_weight.shape].append(reshaped_weight)
    return {shape: tensors for shape, tensors in groups.items() if len(tensors) > 1}

def format_bytes(size: int) -> str:
    """Formats the size in bytes."""
    if size == 0: return "0 B"
    power = 1024; n = 0
    labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size >= power and n < len(labels) - 1:
        size /= power; n += 1
    return f"{size:.2f} {labels[n]}"

def analyze_model(model_name: str):
    """
    Performs a full analysis cycle for the specified model with maximum compression and a full report.
    """
    print("\n" + "="*80)
    print(f"RUNNING A FULL ANALYSIS FOR THE MODEL: {model_name.upper()}")
    print("="*80)

    if model_name.lower() == "gpt-2":
        model = GPT2Model.from_pretrained('gpt2')
        layer_type = Conv1D
    elif model_name.lower() == "resnet-18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        layer_type = nn.Conv2d
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
        
    weight_groups = group_model_weights(model, layer_type)
    
    if not weight_groups:
        print(f"No layer groups of type {layer_type.__name__} were found in model {model_name} for compression.")
        return
        
    print(f"{len(weight_groups)} weight group(s) found for analysis.\n")
    
    compressor = GABE()
    
    for i, (shape, weights_list) in enumerate(weight_groups.items()):
        num_tensors = len(weights_list)
        print("-" * 80)
        print(f"ANALYSIS OF GROUP {i+1}/{len(weight_groups)}: {num_tensors} tensors of shape {shape}")
        print("-" * 80)
        
        original_stacked = torch.stack(weights_list)
        original_size = original_stacked.numel() * original_stacked.element_size()

        compressed_data = compressor.compress(weights_list, basis_rank=1, w_bar_rank=16)

        w_bar_f, w_bar_r, coeffs, basis_f, basis_r, _ = compressed_data.values()
        
        size_w_bar_f_f16 = sum(t.numel() * 2 for t in w_bar_f)
        size_w_bar_r_int8 = w_bar_r.numel() * 1
        size_coeffs_f16 = coeffs.numel() * 2
        size_basis_f_f16 = sum(t.numel() * 2 for t in basis_f) if basis_f else 0
        size_basis_r_int8 = basis_r.numel() * 1
        
        total_compressed_size = (size_w_bar_f_f16 + size_w_bar_r_int8 + size_coeffs_f16 +
                                 size_basis_f_f16 + size_basis_r_int8)

        print(f"Initial group size (float32): {format_bytes(original_size)}")
        print("\n--- Compressed Data Components (Storage Simulation) ---")
        print("  General part (for the whole group):")
        print(f"    - Formula for w_bar (f16):      {format_bytes(size_w_bar_f_f16)}")
        print(f"    - Remainder for w_bar (int8):   {format_bytes(size_w_bar_r_int8)}")
        print(f"    - Formulas for the Basis (f16): {format_bytes(size_basis_f_f16)}")
        print(f"    - Remainder for Basis (int8):   {format_bytes(size_basis_r_int8)}")
        print("  Individual part (for each tensor):")
        print(f"    - Coefficients (float16):       {format_bytes(size_coeffs_f16)} (всего)")
        print(f"    - For 1 tensor:                 {format_bytes(size_coeffs_f16 / num_tensors)}")

        print("\n--- Compression results ---")
        print(f"Final compressed size:     {format_bytes(total_compressed_size)}")
        print(f"Compression ratio:         {original_size / total_compressed_size:.2f}x")

        reconstructed_weights = compressor.decompress(compressed_data)
        stacked_rec = torch.stack(reconstructed_weights)
        
        diff = original_stacked - stacked_rec
        mse = torch.mean(diff.pow(2)).item()
        rmse = math.sqrt(mse)
        mae = torch.mean(torch.abs(diff)).item()
        signal_var = torch.var(original_stacked).item()
        snr = 10 * math.log10(signal_var / mse) if mse > 1e-30 else float('inf')

        print("\n--- Verification of recovery accuracy ---")
        print(f"  - MSE (Mean Squared Error):  {mse:.3e}")
        print(f"  - RMSE (Root MSE):           {rmse:.3e}")
        print(f"  - MAE (Mean Absolute Error): {mae:.3e}")
        print(f"  - SNR (Signal-to-Noise):     {snr:.1f} dB\n")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    analyze_model("GPT-2")
    analyze_model("ResNet-18")