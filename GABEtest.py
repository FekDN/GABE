# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

import torch
import torch.nn as nn
from typing import List, Dict
import math
import warnings
from collections import defaultdict

# --- Local Import ---
from GABE import GABE

# --- Models for Analysis ---
from transformers import AutoModel, Conv1D
from torchvision.models import resnet18, ResNet18_Weights
from diffusers import StableDiffusionPipeline

def format_bytes(size: int) -> str:
    """Formats size in bytes into a human-readable string."""
    if size == 0: return "0 B"
    power = 1024; n = 0
    labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size >= power and n < len(labels) - 1:
        size /= power; n += 1
    return f"{size:.2f} {labels[n]}"

def group_model_weights(model: nn.Module, layer_types: List[type]) -> Dict[tuple, List[torch.Tensor]]:
    """Finds and groups weights of specified layer types in a model by their shape."""
    groups = defaultdict(list)
    for module in model.modules():
        if isinstance(module, tuple(layer_types)):
            weight = module.weight.detach().clone()
            
            if isinstance(module, Conv1D):
                reshaped_weight = weight.T
            elif isinstance(module, nn.Conv2d):
                reshaped_weight = weight.view(weight.shape[0], -1)
            elif isinstance(module, nn.Linear):
                reshaped_weight = weight
            else:
                continue

            groups[reshaped_weight.shape].append(reshaped_weight)
    
    return {shape: tensors for shape, tensors in groups.items() if len(tensors) > 1}

def analyze_model_groups(model_name_str: str, model_obj: nn.Module, layer_types: List[type]):
    """Analyzes a single model object and reports detailed results."""
    print(f"\nAnalyzing model component: {model_name_str}")
    print("-" * 80)
    
    weight_groups = group_model_weights(model_obj, layer_types)
    
    if not weight_groups:
        print(f"No compressible groups of layer types {[t.__name__ for t in layer_types]} found.")
        return
        
    print(f"Found {len(weight_groups)} weight group(s) for analysis.\n")
    
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

        print(f"Original group size (float32):     {format_bytes(original_size)}")
        print("\n--- Compressed Data Components (Storage Simulation) ---")
        print("  Shared Part (for the whole group):")
        print(f"    - Formula for w_bar (f16):     {format_bytes(size_w_bar_f_f16)}")
        print(f"    - Residual for w_bar (int8):   {format_bytes(size_w_bar_r_int8)}")
        print(f"    - Formulas for Basis (f16):    {format_bytes(size_basis_f_f16)}")
        print(f"    - Residuals for Basis (int8):  {format_bytes(size_basis_r_int8)}")
        print("  Per-Tensor Part (for each tensor):")
        print(f"    - Coefficients (float16):      {format_bytes(size_coeffs_f16)} (total)")
        print(f"    - Per tensor:                  {format_bytes(size_coeffs_f16 / num_tensors)}")

        print("\n--- Compression Summary ---")
        print(f"Final compressed size:     {format_bytes(total_compressed_size)}")
        print(f"Compression Ratio:         {original_size / total_compressed_size:.2f}x")

        reconstructed_weights = compressor.decompress(compressed_data)
        stacked_rec = torch.stack(reconstructed_weights)
        
        diff = original_stacked - stacked_rec
        mse = torch.mean(diff.pow(2)).item()
        rmse = math.sqrt(mse)
        mae = torch.mean(torch.abs(diff)).item()
        signal_var = torch.var(original_stacked).item()
        snr = 10 * math.log10(signal_var / mse) if mse > 1e-30 else float('inf')

        print("\n--- Reconstruction Accuracy Verification ---")
        print(f"  - MSE (Mean Squared Error):  {mse:.3e}")
        print(f"  - RMSE (Root MSE):           {rmse:.3e}")
        print(f"  - MAE (Mean Absolute Error): {mae:.3e}")
        print(f"  - SNR (Signal-to-Noise):     {snr:.1f} dB\n")

def run_benchmark():
    """Loads and analyzes a suite of different models."""
    models_to_test = [
        "gpt2",
        "resnet-18",
        "roberta-base",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "runwayml/stable-diffusion-v1-5",
    ]

    for model_name in models_to_test:
        print("\n" + "#"*80)
        print(f"# BENCHMARKING MODEL: {model_name.upper()}")
        print("#"*80)
        
        try:
            if model_name == "gpt2":
                model = AutoModel.from_pretrained(model_name)
                analyze_model_groups("GPT-2", model, [Conv1D])
            
            elif model_name == "resnet-18":
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                analyze_model_groups("ResNet-18", model, [nn.Conv2d])
            
            elif "roberta" in model_name or "distilbert" in model_name:
                model = AutoModel.from_pretrained(model_name)
                analyze_model_groups(model_name, model, [nn.Linear])
            
            elif "stable-diffusion" in model_name:
                print("Loading Stable Diffusion... This may take a while and require significant memory.")
                pipeline = StableDiffusionPipeline.from_pretrained(model_name) # (model_name, torch_dtype=torch.float16)
                
                analyze_model_groups("Stable Diffusion - UNet", pipeline.unet, [nn.Conv2d, nn.Linear])
                analyze_model_groups("Stable Diffusion - VAE", pipeline.vae, [nn.Conv2d, nn.Linear])
                analyze_model_groups("Stable Diffusion - Text Encoder", pipeline.text_encoder, [nn.Linear])
            
        except Exception as e:
            print(f"\nERROR: Could not complete analysis for '{model_name}'.")
            print(f"Reason: {e}\n")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    run_benchmark()
