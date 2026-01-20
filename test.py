# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

# Required libraries:
# pip install torch torchvision scikit-learn numpy tensorly transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import math
import numpy as np

# Optional dependencies for advanced methods
try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    print("Warning: scikit-learn is not installed. Product Quantization (Method 18) will be unavailable.")
    print("To install: pip install scikit-learn numpy")
    MiniBatchKMeans = None

try:
    import tensorly as tl
    from tensorly.decomposition import tucker, tensor_train
    tl.set_backend('pytorch')  # Set the backend for tensorly to PyTorch
except ImportError:
    print("Warning: tensorly is not installed. Tucker (HSVD) and TT-SVD methods will be unavailable.")
    print("To install: pip install tensorly")
    tl = None

# Model-specific layer imports
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    from transformers.modeling_utils import Conv1D
from torchvision.models import resnet18, ResNet18_Weights


def format_bytes(size: int) -> str:
    """Formats a size in bytes into a human-readable string (KB, MB, GB)."""
    if size == 0:
        return "0 B"
    power = 1024
    n = 0
    labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size >= power and n < len(labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {labels[n]}"

# --- SVD Decomposition ---
def eigen_basis_extractor(weights_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts the eigen-basis from a list of weight tensors using SVD.
    Returns the mean tensor, basis vectors, and coefficients.
    """
    dtype = weights_list[0].dtype
    stacked = torch.stack(weights_list)
    w_bar = torch.mean(stacked, dim=0)

    L, d1, d2 = stacked.shape
    flattened = stacked.view(L, -1)
    mean_flat = w_bar.view(-1)

    centered = flattened.to(torch.double) - mean_flat.to(torch.double)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    B_stacked = torch.stack([Vh[k].view(d1, d2) for k in range(Vh.shape[0])])
    coeffs = U * S
    return w_bar, B_stacked.to(dtype), coeffs.to(dtype)

# --- Quantization Utilities ---
def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, int]:
    """Performs uniform affine quantization on a tensor."""
    if bits == 8:
        q_min, q_max = -128, 127
    elif bits == 4:
        q_min, q_max = -8, 7
    else:
        raise ValueError("Only 8-bit and 4-bit quantization are supported.")

    t_min, t_max = tensor.min(), tensor.max()
    scale = (t_max - t_min) / (q_max - q_min)
    if scale < 1e-9:
        scale = 1.0
    
    zero_point = int(torch.round(q_min - t_min / scale).clamp(q_min, q_max).item())
    
    quantized_tensor = torch.round(tensor / scale + zero_point).clamp(q_min, q_max).to(torch.int8)
    return quantized_tensor, scale, zero_point

def dequantize_tensor(q_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """De-quantizes a tensor."""
    return (q_tensor.float() - zero_point) * scale

# --- Compression & Decompression Functions ---

def compress_decompress_block_fp8(tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, int]:
    """Simulates Block-FP8 compression and decompression."""
    original_shape = tensor.shape
    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.numel()

    # Pad to be divisible by block_size
    rem = num_elements % block_size
    if rem != 0:
        padding = torch.zeros(block_size - rem, dtype=tensor.dtype, device=tensor.device)
        flat_tensor = torch.cat([flat_tensor, padding])

    # Reshape into blocks
    blocked = flat_tensor.view(-1, block_size)
    
    # Calculate per-block scales
    scales = torch.max(torch.abs(blocked), dim=1, keepdim=True).values
    scales[scales < 1e-10] = 1.0  # Avoid division by zero

    # Simulate quantization and dequantization
    scaled_blocked = blocked / scales
    quantized_sim = torch.round(scaled_blocked * 127) / 127
    dequantized_blocked = quantized_sim * scales
    
    # Reshape back to original
    reconstructed_flat = dequantized_blocked.flatten()
    reconstructed_tensor = reconstructed_flat[:num_elements].view(original_shape)
    
    # Calculate size: FP8 data + float16 scales
    size = num_elements * 1 + scales.numel() * 2
    return reconstructed_tensor, size

def compress_decompress_grouped_quant(tensor: torch.Tensor, group_size: int = 64, bits: int = 4) -> Tuple[torch.Tensor, int]:
    """Performs Grouped Quantization (GQ)."""
    original_shape = tensor.shape
    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.numel()

    # Pad to be divisible by group_size
    rem = num_elements % group_size
    if rem != 0:
        padding = torch.zeros(group_size - rem, dtype=tensor.dtype, device=tensor.device)
        flat_tensor = torch.cat([flat_tensor, padding])
    
    grouped = flat_tensor.view(-1, group_size)
    reconstructed_groups = []
    
    for group in grouped:
        q_group, scale, zp = quantize_tensor(group, bits=bits)
        reconstructed_groups.append(dequantize_tensor(q_group, scale, zp))
    
    reconstructed_flat = torch.cat(reconstructed_groups)
    reconstructed_tensor = reconstructed_flat[:num_elements].view(original_shape)
    
    # Size: data (e.g., 4 bits/elem) + metadata per group (scale+zp = 8 bytes)
    num_groups = grouped.shape[0]
    size = num_elements * (bits / 8) + num_groups * 8
    return reconstructed_tensor, size

def compress_decompress_pruning(tensor: torch.Tensor, sparsity: float = 0.5) -> Tuple[torch.Tensor, int]:
    """Performs magnitude pruning and calculates sparse COO storage size."""
    flat_tensor = tensor.flatten()
    k = int(sparsity * flat_tensor.numel())
    if k >= flat_tensor.numel():
        return torch.zeros_like(tensor), 0
        
    threshold = torch.kthvalue(torch.abs(flat_tensor), k).values.item()
    mask = torch.abs(tensor) > threshold
    reconstructed_tensor = tensor * mask
    
    # Size: non-zero values (float16) + their indices (int32)
    num_non_zero = torch.sum(mask).item()
    size_of_value = 2  # float16
    size_of_index = 4  # int32
    size = num_non_zero * (size_of_value + size_of_index)
    return reconstructed_tensor, size

def compress_decompress_pq(tensor: torch.Tensor, num_subvectors: int = 16, num_centroids: int = 256) -> Tuple[torch.Tensor, int]:
    """Performs Product Quantization (PQ) using K-Means."""
    if MiniBatchKMeans is None:
        return tensor, tensor.numel() * 4
        
    original_shape = tensor.shape
    flat_tensor = tensor.flatten().to(torch.float32)
    num_elements = flat_tensor.numel()

    subvector_dim = num_elements // num_subvectors
    if subvector_dim == 0:
        return torch.zeros_like(tensor), 0

    rem = num_elements % num_subvectors
    if rem != 0:
        padding_size = num_subvectors - rem
        padding = torch.zeros(padding_size, dtype=flat_tensor.dtype, device=flat_tensor.device)
        flat_tensor = torch.cat([flat_tensor, padding])
        num_elements = flat_tensor.numel()
        subvector_dim = num_elements // num_subvectors
        
    data = flat_tensor.view(num_subvectors, subvector_dim).t().cpu().numpy()
    kmeans = MiniBatchKMeans(n_clusters=num_centroids, init='k-means++', n_init='auto', batch_size=256, random_state=42)
    kmeans.fit(data)
    codes = kmeans.predict(data)
    
    codebook = torch.from_numpy(kmeans.cluster_centers_).to(tensor.device)
    reconstructed_data = codebook[codes]
    reconstructed_flat = reconstructed_data.t().flatten()
    
    reconstructed_tensor = reconstructed_flat[:original_shape.numel()].view(original_shape)

    # Size: codebook (float16) + codes (1 byte/code for 256 centroids)
    codebook_size = num_centroids * subvector_dim * 2
    codes_size = num_subvectors
    size = codebook_size + codes_size
    return reconstructed_tensor.to(tensor.dtype), size

def compress_decompress_tucker(tensor: torch.Tensor, ranks: List[int]) -> Tuple[torch.Tensor, int]:
    """Performs Tucker Decomposition (HSVD)."""
    if tl is None: return tensor, tensor.numel() * 4
    
    core, factors = tucker(tensor, rank=ranks, init='random', tol=1e-5, n_iter_max=100)
    reconstructed_tensor = tl.tucker_to_tensor((core, factors))
    
    # Size: core tensor + 3 factor matrices in float16
    size = core.numel() * 2 + sum(f.numel() * 2 for f in factors)
    return reconstructed_tensor, size

def compress_decompress_tt(tensor: torch.Tensor, ranks: List[int]) -> Tuple[torch.Tensor, int]:
    """Performs Tensor Train (TT-SVD) Decomposition."""
    if tl is None: return tensor, tensor.numel() * 4

    tensor_cpu = tensor.cpu()
    cores = tensor_train(tensor_cpu, rank=ranks)
    reconstructed_tensor_cpu = tl.tt_to_tensor(cores)
    reconstructed_tensor = reconstructed_tensor_cpu.to(tensor.device)
    
    # Size: sum of all cores in float16
    cores_on_device = [c.to(tensor.device) for c in cores]
    size = sum(c.numel() * 2 for c in cores_on_device)
    return reconstructed_tensor, size

def get_sparse_size(tensor: torch.Tensor) -> int:
    """Calculates the size of a tensor in sparse COO format."""
    num_non_zero = torch.count_nonzero(tensor).item()
    size_of_value = 2  # float16
    size_of_index = 4  # int32
    return num_non_zero * (size_of_value + size_of_index)

def compress_decompress_sparse_svd(weights_list: List[torch.Tensor], sparsity: float = 0.5) -> Tuple[torch.Tensor, int]:
    """A proxy for Sparse SVD: SVD followed by component pruning."""
    num_layers = len(weights_list)
    if num_layers <= 1:
        tensor = weights_list[0]
        k = int(sparsity * tensor.numel())
        threshold = torch.kthvalue(torch.abs(tensor.flatten()), k).values
        pruned_tensor = tensor * (torch.abs(tensor) > threshold)
        return pruned_tensor.unsqueeze(0), get_sparse_size(pruned_tensor)

    w_bar, B, coeffs = eigen_basis_extractor(weights_list)
    
    # Prune each component
    w_bar_flat = w_bar.flatten()
    k_w = int(sparsity * w_bar_flat.numel())
    threshold_w = torch.kthvalue(torch.abs(w_bar_flat), k_w).values if k_w < w_bar_flat.numel() else float('inf')
    w_bar_pruned = w_bar * (torch.abs(w_bar) > threshold_w)

    B_pruned_list = []
    for b_tensor in B:
        b_flat = b_tensor.flatten()
        k_b = int(sparsity * b_flat.numel())
        threshold_b = torch.kthvalue(torch.abs(b_flat), k_b).values if k_b < b_flat.numel() else float('inf')
        B_pruned_list.append(b_tensor * (torch.abs(b_tensor) > threshold_b))
    B_pruned = torch.stack(B_pruned_list)
            
    coeffs_flat = coeffs.flatten()
    k_c = int(sparsity * coeffs_flat.numel())
    threshold_c = torch.kthvalue(torch.abs(coeffs_flat), k_c).values if k_c < coeffs_flat.numel() else float('inf')
    coeffs_pruned = coeffs * (torch.abs(coeffs) > threshold_c)

    # Calculate total sparse size
    total_size = get_sparse_size(w_bar_pruned) + get_sparse_size(B_pruned) + get_sparse_size(coeffs_pruned)
    
    # Reconstruct from pruned components
    K = num_layers - 1
    d1, d2 = w_bar.shape
    B_pruned_k = B_pruned[:K]
    coeffs_pruned_k = coeffs_pruned[:, :K]

    reconstructed_flat = w_bar_pruned.view(1, -1) + torch.matmul(coeffs_pruned_k, B_pruned_k.view(K, -1))
    reconstructed_tensor = reconstructed_flat.view(num_layers, d1, d2)
    
    return reconstructed_tensor, total_size

def compress_decompress_delta_svd(delta: torch.Tensor, k_delta: int) -> Tuple[torch.Tensor, int]:
    """Compresses and decompresses a delta tensor using low-rank SVD."""
    w_bar_d, B_d, coeffs_d = eigen_basis_extractor([t for t in delta])
    K = min(k_delta, B_d.shape[0], coeffs_d.shape[1])
    
    if K == 0:
        reconstructed_delta = w_bar_d.unsqueeze(0).repeat(delta.shape[0], 1, 1)
        return reconstructed_delta, w_bar_d.numel() * 2
        
    B_d_k = B_d[:K]
    coeffs_d_k = coeffs_d[:, :K]
    
    reconstructed_flat = w_bar_d.view(1, -1) + torch.matmul(coeffs_d_k, B_d_k.view(K, -1))
    reconstructed_delta = reconstructed_flat.view(delta.shape)
    
    size = (w_bar_d.numel() + B_d_k.numel() + coeffs_d_k.numel()) * 2
    return reconstructed_delta, size

# --- Main Comparison Function ---

def compare_methods(group_name: str, weights_list: List[torch.Tensor]):
    """
    Compares 21 different compression methods on a given group of weight tensors.
    """
    print(f"\n{'#'*25} COMPREHENSIVE METHOD COMPARISON: {group_name} {'#'*25}")
    
    original_stacked = torch.stack(weights_list)
    num_layers, d1, d2 = original_stacked.shape
    original_size_bytes = original_stacked.numel() * original_stacked.element_size()
    print(f"Original Size ({num_layers} tensors): {format_bytes(original_size_bytes)}")
    
    signal_variance = torch.var(original_stacked)
    
    header = f"{'Method':<40} | {'Size':9} | {'Ratio':9} | {'MSE':9} | {'RMSE':9} | {'MAE':9} | {'SNR(dB)':9}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    # --- Pre-computation for multiple methods ---
    K_opt = num_layers - 1
    K_agg = 1
    meta_size = 8  # assume 8 bytes for scale/zp metadata

    # SVD components
    w_bar_svd, B_svd, coeffs_svd = eigen_basis_extractor(weights_list)
    
    # Pre-quantized SVD components (int8) for hybrid methods
    w_bar_q8, w_bar_s8, w_bar_z8 = quantize_tensor(w_bar_svd, bits=8)
    B_q8_list = [quantize_tensor(B_svd[i], bits=8)[0] for i in range(K_opt)]
    coeffs_q8, coeffs_s8, coeffs_z8 = quantize_tensor(coeffs_svd[:, :K_opt], bits=8)
    
    svd_int8_size = (w_bar_q8.numel() + sum(b.numel() for b in B_q8_list) + coeffs_q8.numel()) * 1 + (K_opt + 2) * meta_size
    
    w_bar_deq8 = dequantize_tensor(w_bar_q8, w_bar_s8, w_bar_z8)
    B_deq8_list = [dequantize_tensor(B_q8_list[i], *quantize_tensor(B_svd[i], bits=8)[1:]) for i in range(K_opt)]
    coeffs_deq8 = dequantize_tensor(coeffs_q8, coeffs_s8, coeffs_z8)
    
    reconstructed_svd_int8 = (w_bar_deq8.view(1, -1) + torch.matmul(coeffs_deq8, torch.stack(B_deq8_list).view(K_opt, -1))).view(num_layers, d1, d2)
    delta_from_svd_int8 = original_stacked - reconstructed_svd_int8

    # --- Execute and Print Results for Each Method ---

    # Method 1: Direct Quantization (int8)
    rec_list, total_size = [], 0
    for tensor in weights_list:
        q, s, z = quantize_tensor(tensor, bits=8)
        rec_list.append(dequantize_tensor(q, s, z))
        total_size += q.numel() * 1 + meta_size
    reconstructed_direct_q8 = torch.stack(rec_list)
    error = original_stacked - reconstructed_direct_q8
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'1. Direct Quantization (int8)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 2: Direct Quantization (int4)
    rec_list, total_size = [], 0
    for tensor in weights_list:
        q, s, z = quantize_tensor(tensor, bits=4)
        rec_list.append(dequantize_tensor(q, s, z))
        total_size += q.numel() * 0.5 + meta_size
    reconstructed = torch.stack(rec_list)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'2. Direct Quantization (int4)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 3: Direct Quantization (Block-FP8)
    reconstructed, total_size = compress_decompress_block_fp8(original_stacked)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'3. Direct Quantization (Block-FP8)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")
    
    # Method 4: Centered Quantization (w_bar + int8)
    w_bar_plain = torch.mean(original_stacked, dim=0)
    centered_tensors = original_stacked - w_bar_plain
    rec_list, total_size = [], w_bar_plain.numel() * 2 
    for tensor in centered_tensors:
        q, s, z = quantize_tensor(tensor)
        rec_list.append(dequantize_tensor(q, s, z))
        total_size += q.numel() * 1 + meta_size
    reconstructed = torch.stack(rec_list) + w_bar_plain
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'4. Centered Quantization (w_bar+int8)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 5: SVD(int8), K=L-1
    error = original_stacked - reconstructed_svd_int8
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / svd_int8_size
    print(f"{'5. SVD(int8), K=L-1':<40} | {format_bytes(int(svd_int8_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")
    
    # Method 6: SVD(int4), K=L-1
    w_bar_q4, w_bar_s4, w_bar_z4 = quantize_tensor(w_bar_svd, bits=4)
    B_q4_list = [quantize_tensor(B_svd[i], bits=4)[0] for i in range(K_opt)]
    coeffs_q4, coeffs_s4, coeffs_z4 = quantize_tensor(coeffs_svd[:, :K_opt], bits=4)
    total_size = (w_bar_q4.numel() + sum(b.numel() for b in B_q4_list) + coeffs_q4.numel()) * 0.5 + (K_opt + 2) * meta_size
    w_bar_deq4 = dequantize_tensor(w_bar_q4, w_bar_s4, w_bar_z4)
    B_deq4_list = [dequantize_tensor(B_q4_list[i], *quantize_tensor(B_svd[i], bits=4)[1:]) for i in range(K_opt)]
    coeffs_deq4 = dequantize_tensor(coeffs_q4, coeffs_s4, coeffs_z4)
    reconstructed = (w_bar_deq4.view(1, -1) + torch.matmul(coeffs_deq4, torch.stack(B_deq4_list).view(K_opt, -1))).view(num_layers, d1, d2)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'6. SVD(int4), K=L-1':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 7: SVD(int8), K=1
    B_q_k1, s, z = quantize_tensor(B_svd[0], bits=8)
    B_deq_k1 = dequantize_tensor(B_q_k1, s, z)
    coeffs_q_k1, s, z = quantize_tensor(coeffs_svd[:, :K_agg], bits=8)
    coeffs_deq_k1 = dequantize_tensor(coeffs_q_k1, s, z)
    total_size = (w_bar_q8.numel() + B_q_k1.numel() + coeffs_q_k1.numel()) * 1 + 3 * meta_size
    reconstructed = (w_bar_deq8.view(1, -1) + torch.matmul(coeffs_deq_k1, B_deq_k1.view(K_agg, -1))).view(num_layers, d1, d2)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'7. SVD(int8), K=1':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 8: SVD(float16), K=L-1
    reconstructed = (w_bar_svd.view(1, -1) + torch.matmul(coeffs_svd[:, :K_opt], B_svd[:K_opt].view(K_opt, -1))).view(num_layers, d1, d2)
    total_size = (w_bar_svd.numel() + B_svd[:K_opt].numel() + coeffs_svd[:, :K_opt].numel()) * 2
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'8. SVD(float16), K=L-1':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 9: SVD(float16), K=1
    reconstructed = (w_bar_svd.view(1, -1) + torch.matmul(coeffs_svd[:, :K_agg], B_svd[:K_agg].view(K_agg, -1))).view(num_layers, d1, d2)
    total_size = (w_bar_svd.numel() + B_svd[:K_agg].numel() + coeffs_svd[:, :K_agg].numel()) * 2
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'9. SVD(float16), K=1':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 10: SVD(int8) + Delta(float16)
    delta = (original_stacked - reconstructed_svd_int8).to(torch.float16)
    reconstructed = reconstructed_svd_int8 + delta.to(reconstructed_svd_int8.dtype)
    total_size = svd_int8_size + delta.numel() * 2
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'10. SVD(int8) + Delta(float16)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse) if mse > 1e-20 else float('inf'):9.1f}")

    # Method 11: Quant(int8) + Delta(float16)
    delta = (original_stacked - reconstructed_direct_q8).to(torch.float16)
    reconstructed = reconstructed_direct_q8 + delta.to(reconstructed_direct_q8.dtype)
    total_size = (reconstructed_direct_q8.numel() * 1 + num_layers * meta_size) + delta.numel() * 2
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'11. Quant(int8) + Delta(float16)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse) if mse > 1e-20 else float('inf'):9.1f}")
    
    # Method 12: SVD(int8) + Delta(Block-FP8)
    reconstructed_delta, size_delta = compress_decompress_block_fp8(delta_from_svd_int8)
    reconstructed = reconstructed_svd_int8 + reconstructed_delta
    total_size = svd_int8_size + size_delta
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'12. SVD(int8) + Delta(Block-FP8)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 13: SVD(int8) + Delta(SVD, K=1)
    reconstructed_delta, size_delta = compress_decompress_delta_svd(delta_from_svd_int8.to(torch.float32), k_delta=1)
    reconstructed = reconstructed_svd_int8 + reconstructed_delta
    total_size = svd_int8_size + size_delta
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'13. SVD(int8) + Delta(SVD, K=1)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 14: (SVD-Denoised) + Quant(int8) - identical to Method 1
    # print(f"{'14. (SVD-Denoised) + Quant(int8)':<40} | ...")
    
    # Method 15: Quant->SVD(int8) + Delta(float16)
    w_bar_q, B_q, coeffs_q = eigen_basis_extractor([t for t in reconstructed_direct_q8])
    K_q = len(reconstructed_direct_q8) - 1
    w_bar_qq, s, z = quantize_tensor(w_bar_q)
    B_qq_list = [quantize_tensor(B_q[i])[0] for i in range(K_q)]
    coeffs_qq, _, _ = quantize_tensor(coeffs_q[:, :K_q])
    w_bar_deqq = dequantize_tensor(w_bar_qq, s, z)
    B_deqq_list = [dequantize_tensor(B_qq_list[i], *quantize_tensor(B_q[i])[1:]) for i in range(K_q)]
    coeffs_deqq = dequantize_tensor(coeffs_qq, *quantize_tensor(coeffs_q[:,:K_q])[1:])
    reconstructed_approx_q = (w_bar_deqq.view(1, -1) + torch.matmul(coeffs_deqq, torch.stack(B_deqq_list).view(K_q, -1))).view(num_layers, d1, d2)
    delta = (original_stacked - reconstructed_approx_q).to(torch.float16)
    reconstructed = reconstructed_approx_q + delta.to(reconstructed_approx_q.dtype)
    base_size = (w_bar_qq.numel() + sum(b.numel() for b in B_qq_list) + coeffs_qq.numel()) * 1 + (K_q + 2) * meta_size
    total_size = base_size + delta.numel() * 2
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size
    print(f"{'15. Quant->SVD(int8) + Delta(float16)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse) if mse > 1e-20 else float('inf'):9.1f}")

    # Method 16: Grouped Quantization (GQ-int4)
    rec_list, total_size = [], 0
    for tensor in weights_list:
        rec, size = compress_decompress_grouped_quant(tensor, group_size=64, bits=4)
        rec_list.append(rec)
        total_size += size
    reconstructed = torch.stack(rec_list)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size if total_size > 0 else 0
    print(f"{'16. Grouped Quantization (GQ-int4, G=64)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")
    
    # Method 17: Magnitude Pruning + Sparse COO
    SPARSITY = 0.5
    reconstructed, total_size = compress_decompress_pruning(original_stacked, sparsity=SPARSITY)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size if total_size > 0 else 0
    print(f"{f'17. Pruning {int(SPARSITY*100)}% + Sparse COO':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")
    
    # Method 18: Product Quantization (PQ)
    if MiniBatchKMeans is not None:
        rec_list, total_size = [], 0
        NUM_SUBVECTORS = d2 // 4 if d2 // 4 > 0 else 1
        NUM_CENTROIDS = 256
        for tensor in weights_list:
             rec, size = compress_decompress_pq(tensor, num_subvectors=NUM_SUBVECTORS, num_centroids=NUM_CENTROIDS)
             rec_list.append(rec)
             total_size += size
        reconstructed = torch.stack(rec_list)
        error = original_stacked - reconstructed
        mse = torch.mean(error.pow(2)).item()
        cr = original_size_bytes / total_size if total_size > 0 else 0
        print(f"{f'18. Product Quant (PQ, {NUM_CENTROIDS}c)':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 19: Tucker Decomposition (HSVD)
    if tl:
        TUCKER_RANKS = [num_layers, d1 // 4, d2 // 4]
        reconstructed, total_size = compress_decompress_tucker(original_stacked.clone(), ranks=TUCKER_RANKS)
        error = original_stacked - reconstructed
        mse = torch.mean(error.pow(2)).item()
        cr = original_size_bytes / total_size if total_size > 0 else 0
        print(f"{'19. Tucker (HSVD) R=d/4':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    # Method 20: Tensor Train (TT-SVD)
    if tl:
        TT_RANKS = [1, 16, 16, 1]
        reconstructed, total_size = compress_decompress_tt(original_stacked.clone(), ranks=TT_RANKS)
        error = original_stacked - reconstructed
        mse = torch.mean(error.pow(2)).item()
        cr = original_size_bytes / total_size if total_size > 0 else 0
        print(f"{'20. Tensor Train (TT-SVD) R=16':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")
        
    # Method 21: Sparse SVD (Proxy)
    SPARSITY = 0.8
    reconstructed, total_size = compress_decompress_sparse_svd(weights_list, sparsity=SPARSITY)
    error = original_stacked - reconstructed
    mse = torch.mean(error.pow(2)).item()
    cr = original_size_bytes / total_size if total_size > 0 else 0
    print(f"{f'21. Sparse SVD (Proxy) {int(SPARSITY*100)}%':<40} | {format_bytes(int(total_size)):9} | {cr:8.2f}x | {mse:9.2e} | {math.sqrt(mse):9.2e} | {torch.mean(torch.abs(error)).item():9.2e} | {10 * math.log10(signal_variance / mse):9.1f}")

    print("-" * len(header))


# --- Main Execution Block ---

def run_gpt2_analysis():
    """Loads GPT-2 and analyzes its weight groups."""
    print("="*40)
    print("Analyzing GPT-2 Model")
    print("="*40)
    print("\nLoading pre-trained GPT-2 model...")
    from transformers import GPT2Model
    model = GPT2Model.from_pretrained('gpt2')
    
    print("\nGrouping weights by operation type and shape...")
    operation_groups: Dict[tuple, List[torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            # Weights in GPT-2's Conv1D need to be transposed
            weight_tensor = module.weight.T.detach().clone()
            shape = tuple(weight_tensor.shape)
            if shape not in operation_groups:
                operation_groups[shape] = []
            operation_groups[shape].append(weight_tensor)

    print(f"\nFound {len(operation_groups)} unique operation groups.")
    for shape, weights_list in operation_groups.items():
        if len(weights_list) > 1:  # SVD-based methods require more than one tensor
            role = "Unknown Role"
            if shape == (3072, 768): role = "FFN Layer 1 (Expansion)"
            if shape == (768, 3072): role = "FFN Layer 2 (Projection)"
            if shape == (2304, 768): role = "Attention (QKV Projection)"
            if shape == (768, 768): role = "Attention (Output Projection)"
            group_name = f"{role} (Shape: {shape[0]}x{shape[1]})"
            compare_methods(group_name, weights_list)

def run_resnet18_analysis():
    """Loads ResNet-18 and analyzes its weight groups."""
    print("\n" + "="*40)
    print("Analyzing ResNet-18 Model")
    print("="*40)
    print("\nLoading pre-trained ResNet-18 model...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    print("\nGrouping weights by operation type and shape...")
    operation_groups: Dict[tuple, List[torch.Tensor]] = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_tensor_4d = module.weight.detach().clone()
            
            # Reshape 4D Conv kernel into a 2D matrix for analysis
            # (out_channels, in_channels, kH, kW) -> (out_channels, in_channels * kH * kW)
            num_out, _, _, _ = weight_tensor_4d.shape
            weight_tensor_2d = weight_tensor_4d.view(num_out, -1)
            
            shape = tuple(weight_tensor_2d.shape)
            if shape not in operation_groups: 
                operation_groups[shape] = []
            operation_groups[shape].append(weight_tensor_2d)

    print(f"\nFound {len(operation_groups)} unique operation groups.")
    for shape, weights_list in operation_groups.items():
        if len(weights_list) > 1:
            role = "Unknown Role"
            if shape[1] == 576:   role = "3x3 Convolutions (64 channels)"
            elif shape[1] == 1152: role = "3x3 Convolutions (128 channels)"
            elif shape[1] == 2304: role = "3x3 Convolutions (256 channels)"
            elif shape[1] == 4608: role = "3x3 Convolutions (512 channels)"
                
            group_name = f"{role} (Shape: {shape[0]}x{shape[1]})"
            compare_methods(group_name, weights_list)


if __name__ == "__main__":
    run_gpt2_analysis()
    run_resnet18_analysis()
