# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_ntk.py — Experiment 10: Empirical NTK Alignment
#
# PURPOSE:
#   Tests whether GABE basis directions B_k align with the dominant directions
#   of the Empirical Neural Tangent Kernel (eNTK), restricted to one
#   representative layer of the target group.
#
#   The full eNTK is an (N×C, N×C) matrix over data points and output classes.
#   We work with the FEATURE-SPACE NTK — the (D, D) matrix formed by
#   contracting over data and outputs:
#
#       K_feat = (1/N) Σ_i  J_i^T J_i
#
#   where J_i ∈ R^{C × D} is the Jacobian of model outputs w.r.t. W for sample i.
#   This is the "weight-space" view of the NTK: directions v ∈ R^D along which
#   the model's output changes most across the dataset.
#
#   K_feat-vector product:
#
#       K_feat @ v = (1/N) Σ_i  J_i^T (J_i v)
#
#   J_i v is computed as:  forward-mode JVP  OR  two backward passes.
#   We use the efficient two-pass approach:
#     Step 1: u = J_i v   (via vJp with dummy output dot)
#     Step 2: J_i^T u     (via backprop of Σ_c u_c * f_c(x_i))
#
# WHY NTK:
#   The NTK governs training dynamics in the linearised regime.
#   Its dominant directions are those the model learns fastest.
#   If GABE basis aligns with them, inter-layer variance encodes learning speed.
#
# METRICS:  identical to Experiments 8–9 (A: subspace, B: Rayleigh, C: energy ratio)
# BASELINE: random orthonormal basis, bootstrap n=100
#
# USAGE:
#   python GABEtest_ntk.py
#   python GABEtest_ntk.py --shape 64 64 3 3 --K 3 --n_samples 128 --device cpu
#   Note: NTK computation is O(N × C × D).  Keep n_samples ≤256 on CPU.

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
# Feature-space NTK — vector product
# ---------------------------------------------------------------------------

def build_ntk_mvp(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    param: nn.Parameter,
    device: str = "cpu",
) -> tuple:
    """
    Builds K_feat @ v  =  (1/N) Σ_i  J_i^T (J_i @ v)

    J_i @ v  is a C-dimensional vector (output perturbation for sample i
    when W is perturbed by v).  Computed via forward-mode AD approximation:

        J_i @ v  ≈  (f(x_i, W + ε v) - f(x_i, W)) / ε

    Then J_i^T u  =  ∂(u · f(x_i)) / ∂W  (standard backprop).

    This finite-difference JVP is portable (no functorch required) and
    accurate enough for eigenspace estimation (ε = 1e-4).
    """
    model.eval()
    D = param.numel()
    eps = 1e-4

    # Collect per-sample (J_i @ v) for arbitrary v — done inside the closure
    # by storing all Jacobian rows J_i explicitly as (N, C, D) is too large.
    # Instead we keep the closure that recomputes J_i v on the fly for each v.
    # Precompute the base outputs f(x_i) to avoid double forward passes.
    base_outputs = []   # list of (C,) tensors
    inputs_list  = []   # list of (1, ...) input tensors

    print("  Collecting base outputs for NTK...")
    with torch.no_grad():
        for x_batch, _ in data_loader:
            for i in range(x_batch.size(0)):
                x = x_batch[i:i+1].to(device)
                out = model(x)   # (1, C)
                base_outputs.append(out.squeeze(0).detach())
                inputs_list.append(x.detach())

    N = len(base_outputs)
    C = base_outputs[0].shape[0]

    def ntkfvp(v: torch.Tensor) -> torch.Tensor:
        """K_feat @ v  (D-dimensional)"""
        v_param = v.detach().view_as(param)
        result = torch.zeros(D, device=device)

        for i in range(N):
            x = inputs_list[i]

            # Step 1: J_i @ v  via finite difference
            # Temporarily perturb param
            with torch.no_grad():
                param.data.add_(eps * v_param)
            out_perturbed = model(x).squeeze(0)   # (C,)
            with torch.no_grad():
                param.data.sub_(eps * v_param)

            Jiv = (out_perturbed - base_outputs[i]) / eps   # (C,) — no_grad safe

            # Step 2: J_i^T (J_i v)  via backprop of scalar  Jiv · f(x_i)
            model.zero_grad()
            out = model(x).squeeze(0)                        # (C,) with grad
            scalar = (Jiv.detach() * out).sum()
            scalar.backward()
            g = param.grad.detach().reshape(-1).clone()
            param.grad = None

            result += g

        return result / N

    # Tr(K_feat) ≈ (1/N) Σ_i Tr(J_i^T J_i) = (1/N) Σ_i ||J_i||_F^2
    # Estimate cheaply using Hutchinson on ntkfvp
    print("  Estimating Tr(K_feat) via Hutchinson (16 probes)...")
    trace_K = 0.0
    for _ in range(16):
        z = torch.randn(D, device=device)
        trace_K += (z @ ntkfvp(z)).item()
    trace_K /= 16
    print(f"  N={N}   C={C}   Tr(K_feat) ≈ {trace_K:.6f}")

    return ntkfvp, trace_K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    target_shape=(64, 64, 3, 3),
    K: int = 3,
    n_samples: int = 128,
    n_iter: int = 50,
    n_bootstrap: int = 100,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 10: Empirical NTK Alignment")
    print("=" * 62)
    print(f"Target shape : {target_shape}   K={K}   n_samples={n_samples}")
    print()

    # 1. Model + data
    print("[1/5] Loading model and data...")
    model, _, loader = build_model_and_data(n_samples, batch_size=1, device=device)

    # 2. GABE basis
    print("[2/5] Extracting GABE basis...")
    B_gabe, D, K_actual = extract_gabe_basis(model, target_shape, device)
    if K > K_actual:
        print(f"  WARNING: K={K} > L-1={K_actual}, clamping.")
        K = K_actual
    B_gabe = B_gabe[:, :K]
    print(f"  D={D}   K={K}")

    # 3. NTK MVP
    print("[3/5] Building NTK feature-space MVP...")
    param = get_representative_param(model, target_shape)
    ntkfvp, trace_K = build_ntk_mvp(model, loader, param, device)

    # 4. Top-K NTK eigenvectors
    print(f"[4/5] Top-{K} NTK eigenvectors (power iteration, {n_iter} steps)...")
    V_top, eigvals = top_eigenvectors_via_power_iter(D, K, ntkfvp, n_iter, device)
    print(f"  Top-{K} NTK eigenvalues: {eigvals}")

    # 5. Metrics + bootstrap
    print(f"[5/5] Metrics + bootstrap (n={n_bootstrap})...")
    B_rand = random_orthonormal(D, K, device)
    R_gabe_val = sum((B_gabe[:, i] @ ntkfvp(B_gabe[:, i])).item()
                     for i in range(K)) / (trace_K + 1e-12)
    p_value, null_dist = bootstrap_pvalue(R_gabe_val, D, K, ntkfvp, trace_K,
                                          n_bootstrap, device)

    results = print_results(
        experiment="Exp 10: Empirical NTK Alignment",
        matrix_name="Feature-space NTK  K_feat = (1/N) Σ J_i^T J_i",
        target_shape=target_shape, K=K, D=D,
        trace_M=trace_K, top_eigenvalues=eigvals,
        B_gabe=B_gabe, B_rand=B_rand, V_top=V_top,
        mvp=ntkfvp, p_value=p_value, null_dist=null_dist,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",       type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--K",           type=int, default=3)
    parser.add_argument("--n_samples",   type=int, default=128,
                        help="NTK needs 2 forward+backward passes per sample. "
                             "Keep ≤256 on CPU for tractability.")
    parser.add_argument("--n_iter",      type=int, default=50)
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.K, args.n_samples,
        args.n_iter, args.n_bootstrap, args.device, args.seed)
