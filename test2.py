# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import torchvision.models as models

# ------------------------------------------------------------------------------
# Operator-Centric data structures (unchanged, shape-agnostic)
# ------------------------------------------------------------------------------

@dataclass
class OperatorCentricRepresentation:
    mean_operator: torch.Tensor
    operator_basis: torch.Tensor
    coefficients: Dict[str, torch.Tensor]
    original_shape: tuple

    def __repr__(self):
        return (
            f"OperatorCentricRepresentation(\n"
            f"  original_shape={self.original_shape},\n"
            f"  mean_operator.shape={self.mean_operator.shape},\n"
            f"  operator_basis.shape={self.operator_basis.shape} (K, ...),\n"
            f"  num_layers={len(self.coefficients)},\n"
            f"  rank (K)={self.operator_basis.shape[0]}\n"
            f")"
        )


class OperatorCentricExtractor:
    @staticmethod
    def extract(weights: Dict[str, torch.Tensor], rank: int) -> OperatorCentricRepresentation:
        """
        Perform PCA/SVD-based operator-centric decomposition over a set of tensors
        with identical shapes.
        """
        names = list(weights.keys())
        W_list = list(weights.values())

        if not W_list:
            raise ValueError("Empty weight dictionary")

        original_shape = W_list[0].shape
        K_samples = len(W_list)

        # Flatten each tensor into a vector
        W_flat = torch.stack([W.flatten() for W in W_list], dim=0)

        # Mean operator
        mean_flat = W_flat.mean(dim=0)
        W_centered = W_flat - mean_flat

        # SVD over the operator space
        U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)
        K_rank = min(rank, K_samples)

        basis_flat = Vh[:K_rank]
        coeffs = U[:, :K_rank] * S[:K_rank]

        mean_operator = mean_flat.reshape(original_shape)
        operator_basis = basis_flat.reshape(K_rank, *original_shape)
        coefficients = {name: coeffs[i] for i, name in enumerate(names)}

        return OperatorCentricRepresentation(
            mean_operator=mean_operator,
            operator_basis=operator_basis,
            coefficients=coefficients,
            original_shape=original_shape,
        )


# ------------------------------------------------------------------------------
# Fused operator-centric Conv2D kernel
# ------------------------------------------------------------------------------

def operator_centric_conv2d(x, rep, layer_name, stride=1, padding=1):
    """
    Fused Conv2D kernel using operator-centric decomposition.
    """
    out = F.conv2d(x, rep.mean_operator, stride=stride, padding=padding)
    coeffs_l = rep.coefficients[layer_name]

    for k in range(rep.operator_basis.shape[0]):
        projection = F.conv2d(x, rep.operator_basis[k], stride=stride, padding=padding)
        out += coeffs_l[k] * projection

    return out


def reconstruct_w(rep, layer_name):
    """
    Explicit reconstruction of the original weight tensor from the decomposition.
    """
    W = rep.mean_operator.clone()
    coeffs_l = rep.coefficients[layer_name]

    for k in range(rep.operator_basis.shape[0]):
        W += coeffs_l[k] * rep.operator_basis[k]

    return W


def format_bytes(size: int) -> str:
    power = 1024
    n = 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB"}
    while size >= power and n < len(power_labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"


# ==============================================================================
# Main script: direct analysis of pretrained ResNet-18 convolutional kernels
# ==============================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    # --------------------------------------------------------------------------
    # 1. Load pretrained ResNet-18 and collect Conv2D kernels
    # --------------------------------------------------------------------------

    print("--- Loading pretrained ResNet-18 weights ---")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # Group convolution kernels by shape
    tensors_by_shape = defaultdict(list)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.detach().cpu()

            # We focus on spatial convolutions only (3x3 kernels)
            if weight.ndim != 4 or weight.shape[-1] != 3 or weight.shape[-2] != 3:
                continue

            # Skip downsample projections (structurally unique)
            if "downsample" in name:
                continue

            tensors_by_shape[tuple(weight.shape)].append(
                {"name": name, "tensor": weight}
            )

    print(f"Found {len(tensors_by_shape)} groups of Conv2D kernels by shape.")
    if not tensors_by_shape:
        raise RuntimeError("No suitable convolution kernels found.")

    # --------------------------------------------------------------------------
    # 2. Per-group decomposition, verification, and compression analysis
    # --------------------------------------------------------------------------

    for shape, items in tensors_by_shape.items():
        num_tensors = len(items)
        print(
            f"\n{'='*20} Processing kernel group {shape} "
            f"({num_tensors} tensors) {'='*20}"
        )

        if num_tensors < 2:
            print("Not enough tensors in this group. Skipping.")
            continue

        weights_dict = {item["name"]: item["tensor"] for item in items}

        # ----------------------------------------------------------------------
        # Scenario A: Exact decomposition (full-rank verification)
        # ----------------------------------------------------------------------

        print("\n--- Scenario A: Exact decomposition (verification) ---")
        exact_rank = num_tensors
        rep_exact = OperatorCentricExtractor.extract(weights_dict, rank=exact_rank)

        test_item = items[0]
        W_original = test_item["tensor"]
        W_reconstructed = reconstruct_w(rep_exact, test_item["name"])

        mse_exact = torch.mean((W_original - W_reconstructed) ** 2).item()
        is_correct = torch.allclose(W_original, W_reconstructed, atol=1e-4)

        print(f"Exact reconstruction correct: {is_correct}")
        print(f"MSE: {mse_exact:.2e}")

        # ----------------------------------------------------------------------
        # Scenario B: Low-rank approximation (compression)
        # ----------------------------------------------------------------------

        print("\n--- Scenario B: Low-rank approximation (compression) ---")

        ranks_to_test = [r for r in [1, 2, 4] if r < num_tensors]
        if not ranks_to_test:
            print("No valid low-rank settings for this group.")
            continue

        original_size_bytes = num_tensors * np.prod(shape) * 4
        print(f"Original group size: {format_bytes(original_size_bytes)}")
        print("-" * 70)
        print(f"{'Rank (K)':>10} | {'Compression Ratio':>20} | {'MSE':>15}")
        print("-" * 70)

        for K_approx in ranks_to_test:
            rep_compressed = OperatorCentricExtractor.extract(weights_dict, rank=K_approx)

            mean_size = rep_compressed.mean_operator.nelement() * 4
            basis_size = rep_compressed.operator_basis.nelement() * 4
            coeffs_size = (
                len(rep_compressed.coefficients)
                * rep_compressed.operator_basis.shape[0]
                * 4
            )

            compressed_size = mean_size + basis_size + coeffs_size
            compression_ratio = original_size_bytes / compressed_size

            W_recon_approx = reconstruct_w(rep_compressed, test_item["name"])
            mse = torch.mean((W_original - W_recon_approx) ** 2).item()

            print(f"{K_approx:>10} | {compression_ratio:>19.2f}x | {mse:>14.2e}")

        print("-" * 70)
