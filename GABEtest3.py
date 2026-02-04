# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
# Generating GABE coefficients for stable layers and transfer of "skills"

import torch
import torch.nn as nn
from collections import defaultdict
from GABE import GABE
import timm
from torchvision.models import resnet18, ResNet18_Weights
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
    """Extracting GABE coefficients for all weight groups."""
    weight_groups = group_model_weights(model, layer_types)
    compressor = GABE()
    coeffs_dict = {}
    formulas_dict = {}
    residuals_dict = {}
    for shape, weights_list in weight_groups.items():
        compressed = compressor.compress(weights_list, basis_rank=1, w_bar_rank=16)
        coeffs_dict[shape] = compressed["coeffs"]
        formulas_dict[shape] = (compressed["w_bar_formulas"], compressed["basis_formulas"])
        residuals_dict[shape] = (compressed["w_bar_residuals"], compressed["basis_residuals"])
    return coeffs_dict, formulas_dict, residuals_dict

def compute_correlations(coeffs_dicts: dict):
    """Calculates the correlation of coefficients between two models for each group."""
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
    """Returns a list of layer shapes with correlation >= threshold."""
    stable_shapes = []
    for shape, (_, corr_matrix) in correlations.items():
        # Take the correlation between the first and second models
        corr = corr_matrix[0,1]
        if corr >= threshold:
            stable_shapes.append(shape)
    return stable_shapes

def transfer_coeffs(target_coeffs: dict, source_coeffs: dict, stable_shapes: list):
    """
    Transfer coefficients from source to target for robust layers.
    Returns a new, updated target_coeffs dictionary.
    """
    new_coeffs = target_coeffs.copy()
    for shape in stable_shapes:
        if shape in source_coeffs:
            new_coeffs[shape] = source_coeffs[shape].clone()
    return new_coeffs

def reconstruct_weights_from_coeffs(formulas_dict: dict, residuals_dict: dict, new_coeffs: dict):
    """Recovers tensor weights from new coefficients via GABE."""
    compressor = GABE()
    reconstructed = {}
    for shape in new_coeffs.keys():
        w_bar_formulas, basis_formulas = formulas_dict.get(shape, ((), ()))
        w_bar_residuals, basis_residuals = residuals_dict.get(shape, (torch.empty(0), torch.empty(0)))
        # Decompression of formulas
        w_bar_rec = compressor._decompress_matrix(w_bar_formulas, w_bar_residuals)
        B_rec = compressor._decompress_matrix(basis_formulas, basis_residuals) if basis_residuals.numel() > 0 else torch.empty(0)
        # Tensor recovery
        original_shape = (new_coeffs[shape].shape[0], *w_bar_rec.shape)
        tensors = compressor._reconstruct_weights(w_bar_rec, B_rec, new_coeffs[shape], original_shape)
        reconstructed[shape] = tensors
    return reconstructed

# -----------------------------
# A basic example of "skills" transfer
# -----------------------------

def skill_transfer_example():
    # Loading models
    model_source = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model_target = timm.create_model("hf_hub:SamAdamDay/resnet18_cifar10", pretrained=True)

    layer_types = [nn.Conv2d, nn.Linear]

    # Extracting GABE coefficients and formulas
    source_coeffs, source_formulas, source_residuals = extract_coeffs(model_source, layer_types)
    target_coeffs, target_formulas, target_residuals = extract_coeffs(model_target, layer_types)

    coeffs_dicts = {
        "source": source_coeffs,
        "target": target_coeffs
    }

    # Calculating correlations
    correlations = compute_correlations(coeffs_dicts)

    # Determine stable layers (correlation >= 0.9)
    stable_shapes = identify_stable_layers(correlations, threshold=0.9)
    print("Stable layers for skill transfer:", stable_shapes)

    # Transferring coefficients from source to target for stable layers
    new_target_coeffs = transfer_coeffs(target_coeffs, source_coeffs, stable_shapes)

    # Restore new weights for the target model
    reconstructed_weights = reconstruct_weights_from_coeffs(target_formulas, target_residuals, new_target_coeffs)

    # Checking the sizes of the reconstructed tensors
    for shape, tensors in reconstructed_weights.items():
        print(f"Layer {shape}, number of tensors: {len(tensors)}, shape example: {tensors[0].shape}")

    return reconstructed_weights

# -----------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    reconstructed_weights = skill_transfer_example()
