# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
# Analysis of GABE coefficient patterns for two ResNet‑18 (ImageNet vs CIFAR‑10)

import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

from GABE import GABE

import torchvision
from torchvision.models import resnet18, ResNet18_Weights

import timm

def group_model_weights(model: nn.Module, layer_types: List[type]):
    """
    Collects the weights of the specified layer types into groups by shape.
    Returns a dictionary {shape: [tensor1, tensor2, ...]}.
    """
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


def extract_coeffs_from_model(model: nn.Module, layer_types: List[type]):
    """
    Compresses groups of weights via GABE and returns coefficients: {shape: coeffs_tensor}
    """
    weight_groups = group_model_weights(model, layer_types)
    compressor = GABE()
    coeffs_dict = {}
    for shape, weights_list in weight_groups.items():
        compressed = compressor.compress(weights_list, basis_rank=1, w_bar_rank=16)
        coeffs = compressed["coeffs"]
        coeffs_dict[shape] = coeffs
    return coeffs_dict


def compute_coeff_correlations(models_coeffs: Dict[str, Dict[tuple, torch.Tensor]]):
    """
    Builds correlations of coefficients between models for each general form.
    Returns a dictionary: {shape: (model_names, corr_matrix_numpy)}
    """
    layer_shapes = set()
    for coeffs in models_coeffs.values():
        layer_shapes.update(coeffs.keys())

    correlations = {}
    for shape in layer_shapes:
        model_names = []
        coeff_vectors = []
        for model_name, coeffs_dict in models_coeffs.items():
            if shape in coeffs_dict:
                c = coeffs_dict[shape]
                coeff_vectors.append(c.flatten())
                model_names.append(model_name)

        if len(coeff_vectors) > 1:
            coeff_stack = torch.stack(coeff_vectors)
            corr_matrix = np.corrcoef(coeff_stack.numpy())
            correlations[shape] = (model_names, corr_matrix)

    return correlations


def plot_corr_matrix(model_names, corr_matrix, layer_shape):
    plt.figure(figsize=(6,5))
    sns.heatmap(corr_matrix, xticklabels=model_names, yticklabels=model_names,
                annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Coefficient Correlation — Layer {layer_shape}")
    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN TEST SCENARIO
# -----------------------------

def test_resnet18_imagenet_vs_cifar():
    """
    Loads ResNet‑18 pretrained on ImageNet and ResNet‑18 pretrained on CIFAR‑10,
    extracts GABE coefficients and plots correlations.
    """

    # 1) ResNet‑18 pretrained на ImageNet
    model_imagenet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model_imagenet.eval()

    # 2) ResNet‑18 pretrained на CIFAR‑10 из HF Hub
    model_cifar10 = timm.create_model("hf_hub:SamAdamDay/resnet18_cifar10", pretrained=True)
    model_cifar10.eval()

    models = {
        "ResNet18_ImageNet": model_imagenet,
        "ResNet18_CIFAR10": model_cifar10
    }

    layer_types = [nn.Conv2d, nn.Linear]
    models_coeffs = {}

    # Extracting coefficients
    for name, model in models.items():
        print(f"Extracting coefficients for: {name}")
        coeffs_dict = extract_coeffs_from_model(model, layer_types)
        models_coeffs[name] = coeffs_dict

    # Calculating correlations
    correlations = compute_coeff_correlations(models_coeffs)

    for shape, (model_names, corr_matrix) in correlations.items():
        print(f"\nLayer shape: {shape}")
        print("Correlation matrix:\n", corr_matrix)
        plot_corr_matrix(model_names, corr_matrix, shape)

# -----------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    test_resnet18_imagenet_vs_cifar()
