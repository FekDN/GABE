# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_alignment_utils.py
#
# Shared infrastructure for GABE alignment experiments 9, 10, 11.
# Imported by:
#   GABEtest_fisher.py   — Exp 9:  Fisher Information Matrix alignment
#   GABEtest_ntk.py      — Exp 10: Empirical NTK alignment
#   GABEtest_gradcov.py  — Exp 11: Gradient Covariance Matrix alignment
#
# Each experiment tests a different matrix M and asks:
#   Does the GABE basis B concentrate disproportionate energy in M
#   compared to a random orthonormal basis of identical shape?
#
# All three use the same three metrics (A/B/C) and the same bootstrap procedure
# as Experiment 8 (GABEtest_hessian.py).

import sys, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import percentileofscore
from typing import Callable, Tuple, List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Data / model
# ---------------------------------------------------------------------------

def build_model_and_data(
    n_samples: int = 256,
    batch_size: int = 32,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Module, torch.utils.data.DataLoader]:
    """Pretrained ResNet-18 + small CIFAR-10 subset."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    subset = torch.utils.data.Subset(dataset, list(range(n_samples)))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(device)
    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, loader


# ---------------------------------------------------------------------------
# GABE basis extraction
# ---------------------------------------------------------------------------

def extract_gabe_basis(
    model: nn.Module,
    target_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
        B_ortho : (D_single, K) orthonormal GABE basis
        D_single: numel of one layer
        K       : actual basis rank (= L - 1 for L matching layers)
    """
    gabe = GABE()
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and tuple(module.weight.shape) == target_shape:
            weights.append(module.weight.detach().to(device))

    if len(weights) < 2:
        shapes = sorted({tuple(m.weight.shape)
                         for m in model.modules() if isinstance(m, nn.Conv2d)})
        raise ValueError(
            f"Need ≥2 Conv2d layers with shape {target_shape}, found {len(weights)}.\n"
            f"Available shapes: {shapes}"
        )

    _, B_stacked, _, _ = gabe._extract_svd_components(weights)
    K = B_stacked.shape[0]
    D = B_stacked[0].numel()
    B_flat = B_stacked.view(K, D).T.float()          # (D, K)
    Q, _ = torch.linalg.qr(B_flat)
    return Q[:, :K], D, K


def get_representative_param(
    model: nn.Module,
    target_shape: Tuple[int, ...],
) -> nn.Parameter:
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and tuple(module.weight.shape) == target_shape:
            return module.weight
    raise ValueError(f"No Conv2d with shape {target_shape}")


# ---------------------------------------------------------------------------
# Alignment metrics  (matrix M represented as mvp: v -> M @ v)
# ---------------------------------------------------------------------------

def subspace_alignment(B: torch.Tensor, V: torch.Tensor) -> float:
    """
    Mean squared cosine of principal angles between subspaces B and V.
    B, V: (D, K) orthonormal.  Returns value in [0, 1].
    """
    _, S, _ = torch.linalg.svd(B.T @ V)
    return (S ** 2).mean().item()


def rayleigh_quotients(
    B: torch.Tensor,
    mvp: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """v^T M v for each column of B."""
    return np.array([(B[:, i] @ mvp(B[:, i])).item() for i in range(B.shape[1])])


def energy_ratio(
    B: torch.Tensor,
    mvp: Callable[[torch.Tensor], torch.Tensor],
    trace_M: float,
) -> float:
    """Tr(B^T M B) / Tr(M).  Random baseline: K / D."""
    trace_BtMB = sum((B[:, i] @ mvp(B[:, i])).item() for i in range(B.shape[1]))
    return trace_BtMB / (trace_M + 1e-12)


def random_orthonormal(D: int, K: int, device: str = "cpu") -> torch.Tensor:
    A = torch.randn(D, K, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :K]


def bootstrap_pvalue(
    observed_R: float,
    D: int,
    K: int,
    mvp: Callable[[torch.Tensor], torch.Tensor],
    trace_M: float,
    n_bootstrap: int = 100,
    device: str = "cpu",
) -> Tuple[float, np.ndarray]:
    null = []
    for _ in range(n_bootstrap):
        Q = random_orthonormal(D, K, device)
        null.append(energy_ratio(Q, mvp, trace_M))
    p = 1.0 - percentileofscore(null, observed_R) / 100.0
    return p, np.array(null)


# ---------------------------------------------------------------------------
# Shared report printer
# ---------------------------------------------------------------------------

def print_results(
    experiment: str,
    matrix_name: str,
    target_shape: Tuple,
    K: int,
    D: int,
    trace_M: float,
    top_eigenvalues: np.ndarray,
    B_gabe: torch.Tensor,
    B_rand: torch.Tensor,
    V_top: torch.Tensor,
    mvp: Callable,
    p_value: float,
    null_dist: np.ndarray,
) -> Dict:
    align_gabe = subspace_alignment(B_gabe, V_top)
    align_rand  = subspace_alignment(B_rand,  V_top)
    rq_gabe     = rayleigh_quotients(B_gabe, mvp)
    rq_rand     = rayleigh_quotients(B_rand,  mvp)
    R_gabe      = energy_ratio(B_gabe, mvp, trace_M)
    R_rand      = energy_ratio(B_rand,  mvp, trace_M)
    rand_base   = K / D

    print()
    print("=" * 62)
    print(f"RESULTS  —  {experiment}")
    print("=" * 62)
    print(f"Matrix        : {matrix_name}")
    print(f"Layer shape   : {target_shape}   D={D}   K={K}")
    print(f"Tr(M)         : {trace_M:.4f}")
    print(f"Top-{K} eigenvalues: {top_eigenvalues}")
    print()
    print("(A) Subspace Overlap  [0=orthogonal, 1=identical]")
    print(f"    GABE   : {align_gabe:.6f}")
    print(f"    Random : {align_rand:.6f}")
    print()
    print("(B) Rayleigh Quotients per direction")
    print(f"    GABE   : {rq_gabe}   mean={rq_gabe.mean():.4f}")
    print(f"    Random : {rq_rand}   mean={rq_rand.mean():.4f}")
    print(f"    Ratio  : {rq_gabe.mean() / (rq_rand.mean() + 1e-12):.2f}×")
    print()
    print("(C) Energy Ratio  R = Tr(B^T M B) / Tr(M)")
    print(f"    GABE             : {R_gabe:.6f}")
    print(f"    Random (single)  : {R_rand:.6f}")
    print(f"    Bootstrap null   : {null_dist.mean():.6f} ± {null_dist.std():.6f}")
    print(f"    Random baseline  : {rand_base:.6f}  (= K/D)")
    print(f"    p-value          : {p_value:.4f}  (H₀: GABE = random)")
    print()

    if p_value < 0.01:
        verdict = "SIGNIFICANT (p < 0.01)"
    elif p_value < 0.05:
        verdict = "MARGINAL (0.01 ≤ p < 0.05)"
    else:
        verdict = "NOT SIGNIFICANT (p ≥ 0.05)"
    print(f"Statistical verdict : {verdict}")

    ratio = rq_gabe.mean() / (rq_rand.mean() + 1e-12)
    if align_gabe > 0.5:
        scenario = "A — Strong alignment: GABE ≈ top eigenvectors of M."
    elif ratio > 1.5 and p_value < 0.05:
        scenario = (f"C — Elevated but misaligned: GABE carries {ratio:.2f}× more energy than random (p={p_value:.4f}),\n    "
                    "but subspace overlap ≈ 0. Geometry is real but does not dominate M.")
    else:
        scenario = "B — No significant alignment with M."
    print(f"Scenario            : {scenario}")

    return dict(align_gabe=align_gabe, align_rand=align_rand,
                rq_gabe=rq_gabe, rq_rand=rq_rand,
                R_gabe=R_gabe, p_value=p_value, null_dist=null_dist,
                ratio=ratio, trace_M=trace_M)


# ---------------------------------------------------------------------------
# Power iteration in a given subspace  (shared by all three tests)
# ---------------------------------------------------------------------------

def top_eigenvectors_via_power_iter(
    D: int,
    K: int,
    mvp: Callable[[torch.Tensor], torch.Tensor],
    n_iter: int = 50,
    device: str = "cpu",
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Approximates top-K eigenvectors of the matrix represented by mvp
    using power iteration with deflation.

    Returns:
        V     : (D, K)
        eigvals: (K,) Rayleigh quotients
    """
    eigvecs = []
    for _ in range(K):
        v = torch.randn(D, device=device)
        v /= v.norm()
        for _ in range(n_iter):
            Mv = mvp(v)
            for prev in eigvecs:
                Mv -= (prev @ Mv) * prev
            norm = Mv.norm()
            if norm < 1e-12:
                break
            v = Mv / norm
        eigvecs.append(v.detach())
    V = torch.stack(eigvecs, dim=1)
    eigvals = np.array([(V[:, i] @ mvp(V[:, i])).item() for i in range(K)])
    return V, eigvals


def hutchinson_trace(
    D: int,
    mvp: Callable[[torch.Tensor], torch.Tensor],
    n_probes: int = 64,
    device: str = "cpu",
) -> float:
    """Stochastic trace estimate via Hutchinson estimator."""
    trace = 0.0
    for _ in range(n_probes):
        z = torch.randn(D, device=device)
        trace += (z @ mvp(z)).item()
    return trace / n_probes


# ---------------------------------------------------------------------------
# Spectral percentile analysis
# ---------------------------------------------------------------------------

def spectral_percentile_analysis(
    B_gabe: torch.Tensor,
    mvp: Callable[[torch.Tensor], torch.Tensor],
    D: int,
    n_samples: int = 2000,
    device: str = "cpu",
) -> Dict:
    """
    Builds an empirical CDF of Rayleigh quotients v^T M v over random unit
    vectors v ~ Uniform(S^{D-1}), then reports where each GABE direction sits.

    Returns a dict with:
        rq_random_all   : (n_samples,) array — full empirical distribution
        rq_gabe         : (K,) array — GABE Rayleigh quotients
        percentiles     : (K,) array — CDF rank of each GABE direction (0–100)
        median_random   : float
        p95_random      : float
        p99_random      : float
    """
    # Sample random unit vectors and compute their Rayleigh quotients
    rq_random = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device)
        v /= v.norm()
        rq_random.append((v @ mvp(v)).item())
    rq_random = np.array(rq_random)

    # GABE Rayleigh quotients
    K = B_gabe.shape[1]
    rq_gabe = np.array([(B_gabe[:, i] @ mvp(B_gabe[:, i])).item() for i in range(K)])

    # CDF rank of each GABE direction
    percentiles = np.array([percentileofscore(rq_random, rq) for rq in rq_gabe])

    return dict(
        rq_random_all=rq_random,
        rq_gabe=rq_gabe,
        percentiles=percentiles,
        median_random=float(np.median(rq_random)),
        p95_random=float(np.percentile(rq_random, 95)),
        p99_random=float(np.percentile(rq_random, 99)),
        p_max_random=float(rq_random.max()),
    )


def print_spectral_percentiles(matrix_name: str, result: Dict) -> None:
    rq  = result["rq_gabe"]
    pct = result["percentiles"]
    rd  = result["rq_random_all"]

    print(f"\n  Spectral position — {matrix_name}")
    print(f"  Distribution over {len(rd)} random unit vectors:")
    print(f"    p50  = {result['median_random']:.6f}")
    print(f"    p95  = {result['p95_random']:.6f}")
    print(f"    p99  = {result['p99_random']:.6f}")
    print(f"    max  = {result['p_max_random']:.6f}")
    print(f"  GABE directions:")
    for i, (rq_i, pct_i) in enumerate(zip(rq, pct)):
        bar = "█" * int(pct_i / 5)   # 20-char bar
        print(f"    B_{i+1}: λ = {rq_i:.6f}  →  {pct_i:5.1f}th percentile  {bar}")
    print(f"  Mean GABE percentile: {pct.mean():.1f}th")
