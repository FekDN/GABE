# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_fisher.py — Experiment 9: Fisher Information Matrix Alignment
#
# PURPOSE:
#   Tests whether GABE basis directions B_k align with the top eigenvectors
#   of the Empirical Fisher Information Matrix (eFIM) restricted to one
#   representative layer of the target group.
#
#   The Empirical FIM for a parameter W is:
#
#       F = (1/N) Σ_i  g_i g_i^T
#
#   where g_i = ∂L(x_i, y_i)/∂W  (per-sample gradient, flattened).
#
#   FIM-vector product (no full matrix needed):
#
#       F @ v = (1/N) Σ_i (g_i · v) g_i
#
#   This is the natural gradient geometry: directions of high Fisher energy
#   are those along which the model's predictions change most per unit
#   parameter perturbation, weighted by data distribution.
#
# WHY FIM AFTER HESSIAN:
#   Experiment 8 showed GABE directions carry 3× Hessian energy (p<0.001)
#   but do not span the top Hessian eigenvectors.
#   FIM measures a different notion of sensitivity — distributional rather
#   than loss-landscape curvature — and may align more strongly with GABE,
#   since both are derived from weight statistics across layers/samples.
#
# METRICS:  identical to Experiment 8 (A: subspace overlap, B: Rayleigh, C: energy ratio)
# BASELINE: random orthonormal basis, bootstrap n=100
#
# USAGE:
#   python GABEtest_fisher.py
#   python GABEtest_fisher.py --shape 64 64 3 3 --K 3 --n_samples 512 --device cpu

import torch
import torch.nn as nn
import numpy as np
import argparse

from GABEtest_alignment_utils import (
    build_model_and_data, extract_gabe_basis, get_representative_param,
    random_orthonormal, bootstrap_pvalue, print_results,
    top_eigenvectors_via_power_iter, hutchinson_trace,
)


# ---------------------------------------------------------------------------
# Fisher Information Matrix — vector product
# ---------------------------------------------------------------------------

def build_fisher_mvp(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    param: nn.Parameter,
    device: str = "cpu",
) -> tuple:
    """
    Collects per-sample gradients w.r.t. `param` and returns:
        fvp(v) = (1/N) Σ_i (g_i · v) g_i      [Fisher-vector product]
        trace_F                                   [Hutchinson estimate of Tr(F)]

    Per-sample gradients are computed via the standard trick:
        - forward pass on batch-size-1 inputs
        - or via vmap/functorch if available

    We use the simple loop (batch=1) for correctness and portability.
    The loop is over `n_samples` total samples, not the full dataset.
    """
    model.eval()
    D = param.numel()

    grads = []  # will hold (N, D) tensor of per-sample gradients
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

    def fvp(v: torch.Tensor) -> torch.Tensor:
        """F @ v  via  (1/N) Σ (g_i · v) g_i"""
        dots = G @ v          # (N,)
        return (dots.unsqueeze(1) * G).mean(0)   # (D,)

    # Trace estimate: Tr(F) = (1/N) Σ_i ||g_i||^2
    trace_F = (G ** 2).sum(1).mean().item()

    print(f"  Collected {N} per-sample gradients.  Tr(F) ≈ {trace_F:.4f}")
    return fvp, trace_F, N


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
    print("GABE Experiment 9: Fisher Information Matrix Alignment")
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

    # 3. Fisher MVP
    print("[3/5] Computing per-sample gradients and Fisher MVP...")
    param = get_representative_param(model, target_shape)
    fvp, trace_F, N = build_fisher_mvp(model, loss_fn, loader, param, device)

    # 4. Top-K Fisher eigenvectors
    print(f"[4/5] Top-{K} Fisher eigenvectors (power iteration, {n_iter} steps)...")
    V_top, eigvals = top_eigenvectors_via_power_iter(D, K, fvp, n_iter, device)
    print(f"  Top-{K} Fisher eigenvalues: {eigvals}")

    # 5. Metrics + bootstrap
    print(f"[5/5] Metrics + bootstrap (n={n_bootstrap})...")
    B_rand = random_orthonormal(D, K, device)
    R_gabe_val = sum((B_gabe[:, i] @ fvp(B_gabe[:, i])).item()
                     for i in range(K)) / (trace_F + 1e-12)
    p_value, null_dist = bootstrap_pvalue(R_gabe_val, D, K, fvp, trace_F,
                                          n_bootstrap, device)

    results = print_results(
        experiment="Exp 9: Fisher IM Alignment",
        matrix_name="Empirical Fisher Information Matrix  F = (1/N) Σ g_i g_i^T",
        target_shape=target_shape, K=K, D=D,
        trace_M=trace_F, top_eigenvalues=eigvals,
        B_gabe=B_gabe, B_rand=B_rand, V_top=V_top,
        mvp=fvp, p_value=p_value, null_dist=null_dist,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",       type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--K",           type=int, default=3)
    parser.add_argument("--n_samples",   type=int, default=256,
                        help="Per-sample grad collection is O(N); keep ≤512 on CPU")
    parser.add_argument("--n_iter",      type=int, default=50)
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.K, args.n_samples,
        args.n_iter, args.n_bootstrap, args.device, args.seed)
