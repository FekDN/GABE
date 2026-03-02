# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_gradcov.py — Experiment 11: Gradient Covariance Matrix Alignment
#
# PURPOSE:
#   Tests whether GABE basis directions B_k align with the top eigenvectors
#   of the Gradient Covariance Matrix (GCM), restricted to one representative
#   layer of the target group.
#
#   The GCM is the centered version of the empirical Fisher:
#
#       GCM = (1/N) Σ_i  (g_i - ḡ)(g_i - ḡ)^T
#
#   where  g_i = ∂L(x_i, y_i)/∂W  and  ḡ = (1/N) Σ_i g_i.
#
#   GCM-vector product:
#
#       GCM @ v = (1/N) Σ_i (d_i · v) d_i,   where d_i = g_i - ḡ
#
#   This is identical to empirical Fisher but with centered gradients.
#
# WHY GCM vs FISHER:
#   Fisher = GCM + ḡ ḡ^T.
#   The rank-1 term ḡ ḡ^T dominates when gradients share a common direction
#   (e.g. early training). GCM isolates the *variance* of gradients across
#   samples, which is more directly related to generalisation geometry than
#   the mean gradient direction.
#
#   If GCM aligns more strongly with GABE than Fisher does, it means
#   inter-layer variance captures gradient *diversity* across samples
#   rather than the average update direction.
#
# RELATIONSHIP TO GABE:
#   GABE's basis B_k = top right singular vectors of [ΔW_1, ..., ΔW_L].
#   This is the PCA of weight deviations across layers.
#   GCM is the PCA of gradient deviations across samples.
#   Both are covariance-type decompositions — their alignment tests whether
#   layer-space variation and sample-space gradient variation share structure.
#
# METRICS:  identical to Experiments 8–10 (A: subspace, B: Rayleigh, C: energy ratio)
# BASELINE: random orthonormal basis, bootstrap n=100
#
# USAGE:
#   python GABEtest_gradcov.py
#   python GABEtest_gradcov.py --shape 64 64 3 3 --K 3 --n_samples 256 --device cpu

import torch
import torch.nn as nn
import numpy as np
import argparse

from GABEtest_alignment_utils import (
    build_model_and_data, extract_gabe_basis, get_representative_param,
    random_orthonormal, bootstrap_pvalue, print_results,
    top_eigenvectors_via_power_iter,
)


# ---------------------------------------------------------------------------
# Gradient Covariance Matrix — vector product
# ---------------------------------------------------------------------------

def build_gradcov_mvp(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    param: nn.Parameter,
    device: str = "cpu",
) -> tuple:
    """
    Collects per-sample gradients, centers them, and returns:
        gcmvp(v) = (1/N) Σ_i (d_i · v) d_i      [GCM-vector product]
        trace_GCM                                   [Tr(GCM) = (1/N) Σ ||d_i||^2]

    Also returns:
        mean_grad_norm  — ||ḡ||, useful for diagnosing Fisher vs GCM difference
        var_explained   — fraction of Fisher energy in the centering term ḡ ḡ^T
    """
    model.eval()
    D = param.numel()

    # Collect per-sample gradients g_i
    grads = []
    for x_batch, y_batch in data_loader:
        for i in range(x_batch.size(0)):
            x = x_batch[i:i+1].to(device)
            y = y_batch[i:i+1].to(device)
            model.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            g = param.grad.detach().reshape(-1).clone()
            grads.append(g)
            param.grad = None

    G = torch.stack(grads)   # (N, D)
    N = G.shape[0]

    # Center: d_i = g_i - ḡ
    g_mean = G.mean(0)                  # (D,)
    D_mat  = G - g_mean.unsqueeze(0)    # (N, D)  centered deviations

    mean_grad_norm = g_mean.norm().item()

    # Fisher trace and GCM trace for diagnostics
    trace_F   = (G   ** 2).sum(1).mean().item()
    trace_GCM = (D_mat ** 2).sum(1).mean().item()
    # trace_F = trace_GCM + ||ḡ||^2  (by bias-variance decomposition)
    var_explained = (mean_grad_norm ** 2) / (trace_F + 1e-12)

    print(f"  Collected {N} gradients.")
    print(f"  ||ḡ|| = {mean_grad_norm:.6f}   "
          f"Tr(F) = {trace_F:.4f}   Tr(GCM) = {trace_GCM:.4f}")
    print(f"  Mean direction accounts for {100*var_explained:.1f}% of Fisher trace.")

    def gcmvp(v: torch.Tensor) -> torch.Tensor:
        """GCM @ v  =  (1/N) Σ_i (d_i · v) d_i"""
        dots = D_mat @ v              # (N,)
        return (dots.unsqueeze(1) * D_mat).mean(0)   # (D,)

    return gcmvp, trace_GCM, mean_grad_norm, var_explained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    target_shape=(64, 64, 3, 3),
    K: int = 3,
    n_samples: int = 256,
    n_iter: int = 50,
    n_bootstrap: int = 100,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 11: Gradient Covariance Matrix Alignment")
    print("=" * 62)
    print(f"Target shape : {target_shape}   K={K}   n_samples={n_samples}")
    print()

    # 1. Model + data
    print("[1/5] Loading model and data...")
    model, loss_fn, loader = build_model_and_data(n_samples, batch_size=1, device=device)

    # 2. GABE basis
    print("[2/5] Extracting GABE basis...")
    B_gabe, D, K_actual = extract_gabe_basis(model, target_shape, device)
    if K > K_actual:
        print(f"  WARNING: K={K} > L-1={K_actual}, clamping.")
        K = K_actual
    B_gabe = B_gabe[:, :K]
    print(f"  D={D}   K={K}")

    # 3. GCM MVP
    print("[3/5] Computing per-sample gradients and GCM MVP...")
    param = get_representative_param(model, target_shape)
    gcmvp, trace_GCM, mean_grad_norm, var_explained = build_gradcov_mvp(
        model, loss_fn, loader, param, device
    )

    # 4. Top-K GCM eigenvectors
    print(f"[4/5] Top-{K} GCM eigenvectors (power iteration, {n_iter} steps)...")
    V_top, eigvals = top_eigenvectors_via_power_iter(D, K, gcmvp, n_iter, device)
    print(f"  Top-{K} GCM eigenvalues: {eigvals}")

    # 5. Metrics + bootstrap
    print(f"[5/5] Metrics + bootstrap (n={n_bootstrap})...")
    B_rand = random_orthonormal(D, K, device)
    R_gabe_val = sum((B_gabe[:, i] @ gcmvp(B_gabe[:, i])).item()
                     for i in range(K)) / (trace_GCM + 1e-12)
    p_value, null_dist = bootstrap_pvalue(R_gabe_val, D, K, gcmvp, trace_GCM,
                                          n_bootstrap, device)

    results = print_results(
        experiment="Exp 11: Gradient Covariance Alignment",
        matrix_name="Gradient Covariance  GCM = (1/N) Σ (g_i-ḡ)(g_i-ḡ)^T",
        target_shape=target_shape, K=K, D=D,
        trace_M=trace_GCM, top_eigenvalues=eigvals,
        B_gabe=B_gabe, B_rand=B_rand, V_top=V_top,
        mvp=gcmvp, p_value=p_value, null_dist=null_dist,
    )

    # Extra diagnostic: compare Fisher vs GCM result
    print()
    print("── Fisher vs GCM decomposition ──────────────────────────")
    print(f"  Tr(F)   = Tr(GCM) + ||ḡ||²")
    print(f"          = {trace_GCM:.4f}  +  {mean_grad_norm**2:.4f}")
    print(f"  Mean gradient direction carries {100*var_explained:.1f}% of Fisher trace.")
    print("  Run GABEtest_fisher.py and compare R_GABE there vs here")
    print("  to isolate whether GABE aligns with gradient mean or gradient variance.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",       type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--K",           type=int, default=3)
    parser.add_argument("--n_samples",   type=int, default=256,
                        help="Per-sample grad loop. Keep ≤512 on CPU.")
    parser.add_argument("--n_iter",      type=int, default=50)
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.K, args.n_samples,
        args.n_iter, args.n_bootstrap, args.device, args.seed)
