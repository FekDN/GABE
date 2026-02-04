# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

import torch
import torch.nn as nn
import timm
from torchvision.models import resnet18, ResNet18_Weights
from GABE import GABE
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def group_model_weights(model: nn.Module, layer_types: list):
    groups = defaultdict(list)
    for module in model.modules():
        if isinstance(module, tuple(layer_types)):
            weight = module.weight.detach().clone()
            if isinstance(module, nn.Conv2d):
                weight = weight.view(weight.shape[0], -1)
            elif isinstance(module, nn.Linear):
                pass
            else:
                continue
            groups[weight.shape].append(weight)
    return {shape: tensors for shape, tensors in groups.items() if len(tensors) > 1}

def extract_coeffs(model: nn.Module, layer_types: list):
    compressor = GABE()
    weight_groups = group_model_weights(model, layer_types)
    coeffs_dict = {}
    for shape, weights_list in weight_groups.items():
        compressed = compressor.compress(weights_list, basis_rank=1, w_bar_rank=16)
        coeffs_dict[shape] = compressed["coeffs"]
    return coeffs_dict

def compute_correlations(coeffs_dicts: dict):
    correlations = {}
    shapes = set(coeffs_dicts[list(coeffs_dicts.keys())[0]].keys())
    for shape in shapes:
        coeff_vectors = []
        model_names = []
        for model_name, cdict in coeffs_dicts.items():
            if shape in cdict:
                coeff_vectors.append(cdict[shape].flatten())
                model_names.append(model_name)
        if len(coeff_vectors) > 1:
            stack = torch.stack(coeff_vectors)
            corr_matrix = np.corrcoef(stack.numpy())
            correlations[shape] = (model_names, corr_matrix)
    return correlations

def identify_stable_layers(correlations: dict, threshold: float = 0.9):
    stable_shapes = []
    for shape, (_, corr_matrix) in correlations.items():
        corr = corr_matrix[0,1]
        if corr >= threshold:
            stable_shapes.append(shape)
    return stable_shapes

# -----------------------------
# Dependency analysis using multiple batches
# -----------------------------

def analyze_dependency_multi_batch(model, stable_shapes, unstable_shapes, layer_types, num_batches=10, batch_size=8, device='cpu'):
    """
    Collects the coefficients of stable and unstable layers across multiple batches
    and calculates the R² dependence of unstable layers on stable ones.
    """
    compressor = GABE()
    stable_list = []
    unstable_list = []
    
    for _ in range(num_batches):
        # Random batch (can be replaced with real data)
        x = torch.randn(batch_size, 3, 32, 32).to(device)
        coeffs = extract_coeffs(model, layer_types)
        
        # Stable layers
        stable_vec = []
        for s in stable_shapes:
            if s in coeffs:
                stable_vec.append(coeffs[s].flatten())
        stable_list.append(torch.cat(stable_vec))
        
        # Unstable layers
        unstable_vec = []
        for u in unstable_shapes:
            if u in coeffs:
                unstable_vec.append(coeffs[u].flatten())
        unstable_list.append(torch.cat(unstable_vec))
    
    # Forming X and y
    X = torch.stack(stable_list).numpy()      # (num_batches, total_stable_coeffs)
    y = torch.stack(unstable_list).numpy()    # (num_batches, total_unstable_coeffs)
    
    # Linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    return r2

# -----------------------------
# Visualization
# -----------------------------

def plot_r2(values, labels, title):
    plt.figure(figsize=(8,4))
    plt.bar(labels, values)
    plt.ylabel("R²")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main script
# -----------------------------

def dependency_analysis_two_models():
    device = "cpu"
    torch.manual_seed(42)
    
    # Loading two ResNet18 models
    model_source = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model_target = timm.create_model("hf_hub:SamAdamDay/resnet18_cifar10", pretrained=True).to(device)

    layer_types = [nn.Conv2d, nn.Linear]

    # Extracting coefficients to determine stable layers
    source_coeffs = extract_coeffs(model_source, layer_types)
    target_coeffs = extract_coeffs(model_target, layer_types)
    coeffs_dicts = {"source": source_coeffs, "target": target_coeffs}
    
    correlations = compute_correlations(coeffs_dicts)
    stable_shapes = identify_stable_layers(correlations, threshold=0.9)
    unstable_shapes = [s for s in target_coeffs.keys() if s not in stable_shapes]

    print("Stable layers:", stable_shapes)
    print("Unstable layers:", unstable_shapes)

    # Dependency analysis across multiple batches
    r2_source = analyze_dependency_multi_batch(model_source, stable_shapes, unstable_shapes, layer_types,
                                               num_batches=20, batch_size=8, device=device)
    r2_target = analyze_dependency_multi_batch(model_target, stable_shapes, unstable_shapes, layer_types,
                                               num_batches=20, batch_size=8, device=device)

    print("\nR² dependency:")
    print(f"Source model: {r2_source:.4f}")
    print(f"Target model: {r2_target:.4f}")

    # Visualization
    plot_r2([r2_source, r2_target], ["Source", "Target"], "Dependency of unstable layers on stable layers")

# -----------------------------

if __name__ == "__main__":
    dependency_analysis_two_models()
