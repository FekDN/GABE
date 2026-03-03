# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_rmt.py — Experiment 23: Random Matrix Theory Baseline
#
# PURPOSE:
#   Compares the empirical spectral distribution of the GABE weight matrices
#   against the Marchenko-Pastur (MP) distribution — the RMT prediction for
#   random Gaussian matrices with aspect ratio q = n/p.
#
#   If the weight matrix eigenvalue distribution matches MP:
#     → Weights are essentially random; no structure beyond noise.
#     → GABE basis may not capture meaningful geometry.
#
#   If the distribution has a BULK that matches MP plus outlier eigenvalues:
#     → Outliers carry the real signal; GABE B_k should align with outliers.
#     → This is the "spiked covariance model" scenario.
#
#   If the distribution deviates significantly from MP (e.g. power-law bulk):
#     → Weights have non-Gaussian structure; RMT doesn't apply directly.
#
#   METRICS:
#     (a) Empirical eigenvalue density vs MP density (KS test)
#     (b) Number and magnitude of outlier eigenvalues (above MP upper edge)
#     (c) Projection of GABE B_k onto outlier eigenvectors
#
# USAGE:
#   python GABEtest_rmt.py
#   python GABEtest_rmt.py --shape 64 64 3 3 --model resnet18

import sys, os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from scipy.stats import kstest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Marchenko-Pastur density
# ---------------------------------------------------------------------------

def marchenko_pastur_density(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """
    MP density for aspect ratio q = p/n (≤1 assumed; flip if needed).
    q = n_features / n_samples
    lambda_+ = sigma2 * (1 + sqrt(q))^2
    lambda_- = sigma2 * (1 - sqrt(q))^2
    """
    lam_plus  = sigma2 * (1 + np.sqrt(q)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    density = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    xm = x[mask]
    density[mask] = (np.sqrt((lam_plus - xm) * (xm - lam_minus))
                     / (2 * np.pi * q * sigma2 * xm))
    return density


def mp_upper_edge(q: float, sigma2: float = 1.0) -> float:
    return sigma2 * (1 + np.sqrt(q)) ** 2


def fit_sigma2(eigenvalues: np.ndarray, q: float) -> float:
    """Estimate sigma2 by matching the empirical mean to the MP mean = sigma2."""
    return eigenvalues.mean()


# ---------------------------------------------------------------------------
# Subspace projection
# ---------------------------------------------------------------------------

def subspace_alignment(B1, B2):
    _, S, _ = torch.linalg.svd(B1.T @ B2)
    return (S ** 2).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model_name: str = "resnet18",
    target_shapes: list = None,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 23: Random Matrix Theory Baseline")
    print("=" * 62)
    print(f"model={model_name}")
    print()

    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().to(device)
    elif model_name == "vgg11":
        model = torchvision.models.vgg11(weights="IMAGENET1K_V1").eval().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Collect all multi-layer Conv2d groups
    groups = defaultdict(list)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            groups[tuple(m.weight.shape)].append(m.weight.detach())
    groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if target_shapes:
        groups = {k: v for k, v in groups.items() if list(k) in target_shapes}

    print(f"Analyzing {len(groups)} weight groups")
    print()

    all_results = {}

    for shape, ws in sorted(groups.items()):
        L = len(ws)
        print(f"── Shape {shape}  L={L} ───────────────────────")

        # Stack as 2D matrix: (L, D)
        D = ws[0].numel()
        W_mat = torch.stack(ws).view(L, D).float().numpy()   # (L, D)
        n, p = W_mat.shape   # n=L rows, p=D cols
        # Empirical covariance (centered)
        W_c = W_mat - W_mat.mean(0, keepdims=True)
        # Use W^T W / n for eigenvalues (the layer covariance)
        # Shape: (D, D) — too large. Use W W^T / D instead: (L, L) manageable.
        C_small = (W_c @ W_c.T) / D   # (L, L)
        eigvals_small = np.linalg.eigvalsh(C_small)[::-1]   # descending

        # RMT setup: q = L / D  (L ≪ D, so q → 0)
        q = L / D
        sigma2 = fit_sigma2(eigvals_small, q)
        lam_plus = mp_upper_edge(q, sigma2)

        # Count outliers
        n_outliers = int((eigvals_small > lam_plus * 1.05).sum())
        outlier_eigvals = eigvals_small[eigvals_small > lam_plus * 1.05]

        print(f"  Eigenvalue range: [{eigvals_small[-1]:.4f}, {eigvals_small[0]:.4f}]")
        print(f"  MP upper edge   : {lam_plus:.4f}  (q={q:.6f}, σ²={sigma2:.4f})")
        print(f"  Outliers above edge: {n_outliers}  values={outlier_eigvals}")

        # KS test on bulk (excluding outliers)
        bulk = eigvals_small[eigvals_small <= lam_plus * 1.05]
        if len(bulk) > 3:
            x_grid = np.linspace(bulk.min(), bulk.max(), 300)
            mp_cdf = np.cumsum(marchenko_pastur_density(x_grid, q, sigma2))
            mp_cdf /= (mp_cdf[-1] + 1e-12)
            emp_cdf = np.array([np.mean(bulk <= t) for t in x_grid])
            ks_stat = np.max(np.abs(emp_cdf - mp_cdf))
            print(f"  KS(bulk vs MP)  : {ks_stat:.4f}  (0=perfect match, >0.2=poor fit)")
        else:
            ks_stat = np.nan
            print(f"  KS: insufficient bulk eigenvalues")

        # GABE basis
        gabe = GABE()
        _, B_s, _, _ = gabe._extract_svd_components(ws)
        K = B_s.shape[0]
        B_flat = B_s.view(K, D).T.float()
        Q_gabe, _ = torch.linalg.qr(B_flat);  B_gabe = Q_gabe[:, :K]

        # Top eigenvectors of W^T W (from SVD of W_c)
        U, S_sv, _ = np.linalg.svd(W_c, full_matrices=False)
        # Top-K right singular vectors of W_c = top eigenvectors of W_c^T W_c
        V_top = torch.tensor(_.T[:, :K]).float()   # (D, K)
        if V_top.shape[1] < K: V_top = torch.zeros(D, K)

        sa_gabe_top = subspace_alignment(B_gabe, V_top) if V_top.shape[1] == K else float('nan')
        print(f"  Subspace alignment GABE vs top-{K} SVD: {sa_gabe_top:.6f}")

        # Fraction of total spectral energy in GABE directions (W_c proxy)
        W_c_t = torch.tensor(W_c).float()
        trace_total = (W_c_t ** 2).sum().item() / D
        trace_gabe  = sum(
            (B_gabe[:, k].unsqueeze(0) @ W_c_t.T @ W_c_t @ B_gabe[:, k]).item()
            for k in range(K)
        ) / D
        energy_fraction = trace_gabe / (trace_total + 1e-12)
        rand_energy = K / D
        print(f"  Energy fraction GABE: {energy_fraction:.6f}  (random={rand_energy:.6f}, "
              f"ratio={energy_fraction/rand_energy:.2f}×)")
        print()

        all_results[shape] = dict(
            eigvals=eigvals_small, lam_plus=lam_plus, n_outliers=n_outliers,
            ks_stat=ks_stat, sa_top=sa_gabe_top, energy_fraction=energy_fraction,
            rand_energy=rand_energy, q=q
        )

    # --- Summary ---
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"  {'Shape':<24} {'n_out':>6} {'KS':>8} {'sa_top':>8} "
          f"{'E_gabe/rand':>12}")
    print("  " + "-" * 62)
    for shape, r in sorted(all_results.items()):
        ratio = r['energy_fraction'] / (r['rand_energy'] + 1e-12)
        print(f"  {str(shape):<24} {r['n_outliers']:>6} "
              f"{r['ks_stat']:>8.4f} {r['sa_top']:>8.4f} {ratio:>12.2f}×")

    print()
    total_outliers = sum(r['n_outliers'] for r in all_results.values())
    if total_outliers > 0:
        print(f"Outliers detected in {sum(r['n_outliers']>0 for r in all_results.values())} groups.")
        print("Spiked covariance model applies. GABE B_k should align with outlier eigenvectors.")
    else:
        print("No outliers above MP edge. Weight matrices are consistent with random Gaussian.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="resnet18")
    parser.add_argument("--shapes", type=int, nargs="+", action="append", default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run(args.model, args.shapes, args.device, args.seed)
