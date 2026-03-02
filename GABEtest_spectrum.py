# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_spectrum.py — Experiment 12: Spectral Percentile Analysis
#
# PURPOSE:
#   Experiments 8–11 established that GABE directions carry 2–3× more energy
#   than random directions in H, F, and GCM. But "2–3× random" does not say
#   where in the full spectrum GABE sits.
#
#   This experiment answers the precise question:
#
#       In what percentile of the empirical Rayleigh quotient distribution
#       do GABE basis directions fall — for each of the three matrices?
#
#   METHOD:
#     1. Build MVP for each matrix (H, F, GCM) restricted to one representative
#        layer of the target group.
#     2. Sample n_samples random unit vectors, compute v^T M v for each → CDF.
#     3. Compute v^T M v for each GABE direction.
#     4. Report CDF rank (percentile) of each GABE direction.
#
#   This directly resolves "is GABE at the 60th, 80th, or 95th percentile?"
#
# USAGE:
#   python GABEtest_spectrum.py
#   python GABEtest_spectrum.py --shape 64 64 3 3 --K 3 --n_samples 2000 --n_grad 256

import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.autograd import grad as torch_grad

from GABEtest_alignment_utils import (
    build_model_and_data, extract_gabe_basis, get_representative_param,
    spectral_percentile_analysis, print_spectral_percentiles,
)


# ---------------------------------------------------------------------------
# MVP builders (minimal — reuse logic from individual test scripts)
# ---------------------------------------------------------------------------

def build_hessian_mvp(model, loss_fn, loader, param, device):
    """Single-layer Hessian MVP via Pearlmutter trick."""
    model.eval()

    def hvp(v):
        model.zero_grad()
        x, y = next(iter(loader))
        loss = loss_fn(model(x.to(device)), y.to(device))
        g1 = torch_grad(loss, [param], create_graph=True)
        flat_g1 = g1[0].reshape(-1)
        g2 = torch_grad(flat_g1 @ v, [param], retain_graph=True)
        return g2[0].reshape(-1).detach()

    return hvp


def build_fisher_mvp(model, loss_fn, loader, param, device, n_grad):
    """Empirical Fisher MVP: (1/N) sum_i (g_i . v) g_i."""
    model.eval()
    grads = []
    count = 0
    for x_batch, y_batch in loader:
        for i in range(x_batch.size(0)):
            if count >= n_grad:
                break
            x = x_batch[i:i+1].to(device)
            y = y_batch[i:i+1].to(device)
            model.zero_grad()
            loss_fn(model(x), y).backward()
            grads.append(param.grad.detach().reshape(-1).clone())
            param.grad = None
            count += 1
        if count >= n_grad:
            break
    G = torch.stack(grads)  # (N, D)

    def fvp(v):
        dots = G @ v
        return (dots.unsqueeze(1) * G).mean(0)

    trace_F = (G ** 2).sum(1).mean().item()
    return fvp, trace_F


def build_gradcov_mvp(model, loss_fn, loader, param, device, n_grad):
    """Gradient Covariance MVP: centered Fisher."""
    model.eval()
    grads = []
    count = 0
    for x_batch, y_batch in loader:
        for i in range(x_batch.size(0)):
            if count >= n_grad:
                break
            x = x_batch[i:i+1].to(device)
            y = y_batch[i:i+1].to(device)
            model.zero_grad()
            loss_fn(model(x), y).backward()
            grads.append(param.grad.detach().reshape(-1).clone())
            param.grad = None
            count += 1
        if count >= n_grad:
            break
    G = torch.stack(grads)
    g_mean = G.mean(0)
    D_mat = G - g_mean.unsqueeze(0)

    def gcmvp(v):
        dots = D_mat @ v
        return (dots.unsqueeze(1) * D_mat).mean(0)

    trace_GCM = (D_mat ** 2).sum(1).mean().item()
    return gcmvp, trace_GCM


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    target_shape=(64, 64, 3, 3),
    K: int = 3,
    n_spectrum: int = 2000,   # random vectors for CDF
    n_grad: int = 256,        # per-sample gradients for F and GCM
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 12: Spectral Percentile Analysis")
    print("=" * 62)
    print(f"Target shape : {target_shape}   K={K}")
    print(f"CDF samples  : {n_spectrum} random unit vectors per matrix")
    print(f"Grad samples : {n_grad} per-sample gradients (Fisher / GCM)")
    print()

    # 1. Setup
    print("[1/5] Loading model and data...")
    model, loss_fn, loader = build_model_and_data(
        n_samples=n_grad, batch_size=1, device=device
    )

    print("[2/5] Extracting GABE basis...")
    B_gabe, D, K_actual = extract_gabe_basis(model, target_shape, device)
    if K > K_actual:
        print(f"  WARNING: K={K} > L-1={K_actual}, clamping.")
        K = K_actual
    B_gabe = B_gabe[:, :K]
    print(f"  D={D}   K={K}")

    param = get_representative_param(model, target_shape)

    # 2. Build all three MVPs
    print("[3/5] Building matrix MVPs...")
    hvp = build_hessian_mvp(model, loss_fn, loader, param, device)
    fvp, trace_F = build_fisher_mvp(model, loss_fn, loader, param, device, n_grad)
    gcmvp, trace_GCM = build_gradcov_mvp(model, loss_fn, loader, param, device, n_grad)
    print(f"  Tr(F) = {trace_F:.2f}   Tr(GCM) = {trace_GCM:.2f}")

    # 3. Hessian trace estimate (Hutchinson, 32 probes)
    print("  Estimating Tr(H) via Hutchinson...")
    trace_H = 0.0
    for _ in range(32):
        z = torch.randn(D, device=device)
        trace_H += (z @ hvp(z)).item()
    trace_H /= 32
    print(f"  Tr(H) = {trace_H:.2f}")

    # 4. Spectral analysis for each matrix
    print(f"[4/5] Building CDFs ({n_spectrum} random vectors × 3 matrices)...")
    matrices = [
        ("Hessian  (H)",   hvp,   trace_H),
        ("Fisher   (F)",   fvp,   trace_F),
        ("Grad Cov (GCM)", gcmvp, trace_GCM),
    ]

    all_results = {}
    for name, mvp_fn, trace_M in matrices:
        print(f"  {name}  Tr={trace_M:.2f} ...")
        result = spectral_percentile_analysis(B_gabe, mvp_fn, D, n_spectrum, device)
        # Also store normalised RQ: lambda / (Tr(M)/D) — in units of "average eigenvalue"
        avg_eig = trace_M / D
        result["rq_gabe_normalised"] = result["rq_gabe"] / avg_eig
        result["rq_random_p50_normalised"] = result["median_random"] / avg_eig
        all_results[name] = result

    # 5. Report
    print()
    print("[5/5] Results")
    print("=" * 62)

    summary_rows = []
    for name, result in all_results.items():
        print_spectral_percentiles(name, result)
        for i, (rq_i, pct_i, rq_norm_i) in enumerate(zip(
                result["rq_gabe"], result["percentiles"],
                result["rq_gabe_normalised"])):
            summary_rows.append((name, i+1, rq_i, rq_norm_i, pct_i))

    # Summary table
    print()
    print("=" * 62)
    print("SUMMARY TABLE")
    print("=" * 62)
    print(f"{'Matrix':<20} {'Dir':>3}  {'λ_GABE':>10}  {'λ/avg_eig':>10}  {'Percentile':>11}")
    print("-" * 62)
    for row in summary_rows:
        name, i, rq, rq_norm, pct = row
        print(f"{name:<20} B_{i}   {rq:>10.6f}  {rq_norm:>10.2f}×      {pct:>8.1f}th")

    print()
    print("Interpretation guide:")
    print("  λ/avg_eig > 1  → above the mean eigenvalue (expected ~1 for random)")
    print("  Percentile     → CDF rank in empirical distribution of random unit vectors")
    print()

    # Emit the precise claim
    all_pcts = [r["percentiles"] for r in all_results.values()]
    mean_pct_by_matrix = {name: r["percentiles"].mean()
                          for name, r in all_results.items()}
    overall_mean = np.mean([v for v in mean_pct_by_matrix.values()])

    print("Precise positioning of GABE subspace in full spectrum:")
    for name, mean_pct in mean_pct_by_matrix.items():
        print(f"  {name}: mean GABE percentile = {mean_pct:.1f}th")
    print(f"  Overall mean across matrices: {overall_mean:.1f}th percentile")
    print()

    if overall_mean >= 90:
        position = "top decile (≥90th percentile)"
    elif overall_mean >= 75:
        position = "upper quartile (75th–90th percentile)"
    elif overall_mean >= 50:
        position = "mid-to-upper spectrum (50th–75th percentile)"
    else:
        position = "mid-spectrum or below (<50th percentile)"

    print(f"Conclusion: GABE directions sit in the {position}")
    print("of the empirical Rayleigh quotient distribution.")
    print()
    print("This resolves the ambiguity from Experiments 8–11:")
    print("  'elevated vs random' → now quantified as a specific spectral position.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GABE Experiment 12: Spectral Percentile Analysis"
    )
    parser.add_argument("--shape",      type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--K",          type=int, default=3)
    parser.add_argument("--n_spectrum", type=int, default=2000,
                        help="Random unit vectors for empirical CDF (default: 2000)")
    parser.add_argument("--n_grad",     type=int, default=256,
                        help="Per-sample gradients for Fisher/GCM (default: 256)")
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run(
        target_shape=tuple(args.shape),
        K=args.K,
        n_spectrum=args.n_spectrum,
        n_grad=args.n_grad,
        device=args.device,
        seed=args.seed,
    )
