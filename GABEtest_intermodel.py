# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_intermodel.py
# Test: Inter-Model W̄ and Basis Comparison
#
# Two orthogonal comparisons:
#
#   EXPERIMENT A — SAME GROUP, DIFFERENT MODELS
#     For each group shape that exists in both models:
#       Compare W̄_modelA  vs  W̄_modelB   → is "long-term memory" shared?
#       Compare Basis_modelA vs Basis_modelB → is "address space" shared?
#     Sub-cases:
#       A1: Same architecture, same task     (GPT-2 pretrained vs GPT-2 random init)
#       A2: Same architecture, different task (ResNet-18 ImageNet vs ResNet-18 random)
#       A3: Different architecture, matching shapes (GPT-2 vs DistilBERT, shapes ∩)
#
#   EXPERIMENT B — CROSS-GROUP, SAME MODEL  (reference baseline)
#     Within each model: compare W̄ and Basis across groups of different shape.
#     This is the baseline from GABEtest_crossgroup — repeated here for direct
#     comparison so we can answer: "is cross-model similarity > cross-group similarity?"
#
# If A > B  → models share more within same group than groups share within same model
#           → W̄ / Basis are group-specific, not universal across architecture
# If A < B  → groups within a model are more similar than same groups across models
#           → structure is architecture-driven, not training-driven
# If A ≈ B  → both effects are comparably strong

import os, math, warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# GABE core
# ══════════════════════════════════════════════════════════════════════════════

class GABE:
    def decompose(self, weights_list: List[torch.Tensor]):
        if not weights_list:
            raise ValueError("Empty list.")
        dtype = weights_list[0].dtype
        stacked = torch.stack(weights_list)
        shape   = stacked.shape
        w_bar   = stacked.mean(dim=0)
        L = shape[0]
        if L <= 1:
            return w_bar, torch.empty(0, dtype=dtype), torch.empty(0, dtype=dtype), shape
        flat    = stacked.to(torch.float64).view(L, -1)
        centered = flat - w_bar.to(torch.float64).view(-1)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        K = L - 1
        B      = Vh[:K].view(K, *shape[1:]).to(dtype)
        coeffs = (U[:, :K] @ torch.diag(S[:K])).to(dtype)
        return w_bar, B, coeffs, shape


# ══════════════════════════════════════════════════════════════════════════════
# Weight grouping
# ══════════════════════════════════════════════════════════════════════════════

def group_weights(model: nn.Module) -> Dict[tuple, List[torch.Tensor]]:
    """Groups Conv2d / Linear / Conv1D weights by (out, in) shape."""
    groups = defaultdict(list)
    for module in model.modules():
        cname = type(module).__name__
        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().clone()
            w = w.view(w.shape[0], -1)
            groups[tuple(w.shape)].append(w)
        elif isinstance(module, nn.Linear):
            w = module.weight.detach().clone()
            groups[tuple(w.shape)].append(w)
        elif cname == "Conv1D":                 # transformers Conv1D: (in, out)
            w = module.weight.detach().clone().T
            groups[tuple(w.shape)].append(w)
    return {s: t for s, t in groups.items() if len(t) > 1}


# ══════════════════════════════════════════════════════════════════════════════
# Similarity metrics
# ══════════════════════════════════════════════════════════════════════════════

def pearson_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    fa = a.float().flatten().numpy()
    fb = b.float().flatten().numpy()
    n  = min(len(fa), len(fb))
    fa, fb = fa[:n], fb[:n]
    if fa.std() < 1e-10 or fb.std() < 1e-10:
        return 0.0
    r, _ = pearsonr(fa, fb)
    return float(r)

def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)

def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA, invariant to orthogonal transforms and scale."""
    n = min(X.shape[0], Y.shape[0])
    X, Y = _center(X[:n]), _center(Y[:n])
    rank = min(X.shape[1], Y.shape[1], 64)
    _, _, Vx = np.linalg.svd(X, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y, full_matrices=False)
    Xp = X @ Vx[:rank].T
    Yp = Y @ Vy[:rank].T
    def hsic(A, B):
        K = A @ A.T; L = B @ B.T; m = K.shape[0]
        H = np.eye(m) - np.ones((m, m)) / m
        return np.trace(K @ H @ L @ H) / (m - 1) ** 2
    hxy = hsic(Xp, Yp)
    hxx = hsic(Xp, Xp)
    hyy = hsic(Yp, Yp)
    d   = math.sqrt(hxx * hyy)
    return float(hxy / d) if d > 1e-12 else 0.0

def compare_tensors(A: torch.Tensor, B: torch.Tensor) -> Tuple[float, float]:
    """Returns (pearson_r, cka) between two tensors treated as row-matrices."""
    pr  = pearson_flat(A, B)
    Anp = A.float().numpy().reshape(A.shape[0], -1)
    Bnp = B.float().numpy().reshape(B.shape[0], -1)
    cka = cka_linear(Anp, Bnp)
    return pr, cka

def compare_bases(Ba: torch.Tensor, Bb: torch.Tensor) -> Tuple[float, float]:
    """Compare two basis tensors (Ka, Da) and (Kb, Db).
    Pearson: truncate both K and D to min.
    CKA: invariant to D via SVD projection — works across different shapes.
    """
    if Ba.numel() == 0 or Bb.numel() == 0:
        return 0.0, 0.0
    Af = Ba.view(Ba.shape[0], -1).float().numpy()  # (Ka, Da)
    Bf = Bb.view(Bb.shape[0], -1).float().numpy()  # (Kb, Db)
    K  = min(Af.shape[0], Bf.shape[0])
    D  = min(Af.shape[1], Bf.shape[1])             # truncate feature dim too
    rs = [pearsonr(Af[k, :D], Bf[k, :D])[0] for k in range(K)
          if Af[k, :D].std() > 1e-10 and Bf[k, :D].std() > 1e-10]
    pr  = float(np.mean(rs)) if rs else 0.0
    cka = cka_linear(Af, Bf)                       # CKA handles dim mismatch via SVD
    return pr, cka


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — same group, different models
# ══════════════════════════════════════════════════════════════════════════════

def experiment_A(label_A: str, groups_A: Dict,
                 label_B: str, groups_B: Dict) -> dict:
    """
    For every shape present in both models, compare W̄ and Basis.
    Returns dict of results keyed by shape.
    """
    gabe = GABE()
    shared_shapes = sorted(set(groups_A) & set(groups_B))

    print(f"\n{'='*70}")
    print(f"EXPERIMENT A: Same group, different models")
    print(f"  Model A: {label_A}")
    print(f"  Model B: {label_B}")
    print(f"  Shared group shapes: {len(shared_shapes)}")
    print(f"{'='*70}")

    results = {}
    for shape in shared_shapes:
        wA, BsA, _, _ = gabe.decompose(groups_A[shape])
        wB, BsB, _, _ = gabe.decompose(groups_B[shape])

        wbar_pr,  wbar_cka  = compare_tensors(wA, wB)
        basis_pr, basis_cka = compare_bases(BsA, BsB)

        Ka = BsA.shape[0] if BsA.numel() > 0 else 0
        Kb = BsB.shape[0] if BsB.numel() > 0 else 0

        results[shape] = dict(
            wbar_pearson=wbar_pr, wbar_cka=wbar_cka,
            basis_pearson=basis_pr, basis_cka=basis_cka,
            Ka=Ka, Kb=Kb,
            nA=len(groups_A[shape]), nB=len(groups_B[shape]),
        )

        print(f"\n  Group {shape}  "
              f"(A: {len(groups_A[shape])} layers, K={Ka} | "
              f"B: {len(groups_B[shape])} layers, K={Kb})")
        print(f"    W̄   Pearson={wbar_pr:+.3f}   CKA={wbar_cka:.3f}")
        print(f"    Basis Pearson={basis_pr:+.3f}  CKA={basis_cka:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — cross-group, same model (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_B(model_label: str, groups: Dict) -> dict:
    """
    Within one model: compare W̄ and Basis across groups of different shapes.
    Returns aggregated off-diagonal statistics.
    """
    gabe = GABE()
    keys = sorted(groups.keys())
    n    = len(keys)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT B: Cross-group baseline — {model_label}")
    print(f"  Groups: {n}")
    print(f"{'='*70}")

    # Decompose all groups
    wbars, bases = {}, {}
    for shape in keys:
        w, B, _, _ = gabe.decompose(groups[shape])
        wbars[shape] = w
        bases[shape]  = B
        K = B.shape[0] if B.numel() > 0 else 0
        print(f"  {shape}: {len(groups[shape])} layers | K={K}")

    wbar_cka_vals, basis_cka_vals = [], []
    wbar_pr_vals,  basis_pr_vals  = [], []

    for i, si in enumerate(keys):
        for j, sj in enumerate(keys):
            if i >= j:
                continue
            pr_w, cka_w = compare_tensors(wbars[si], wbars[sj])
            pr_b, cka_b = compare_bases(bases[si], bases[sj])
            wbar_pr_vals.append(pr_w);   wbar_cka_vals.append(cka_w)
            basis_pr_vals.append(pr_b);  basis_cka_vals.append(cka_b)

    summary = dict(
        wbar_pearson_mean  = float(np.mean(np.abs(wbar_pr_vals)))  if wbar_pr_vals  else 0,
        wbar_cka_mean      = float(np.mean(wbar_cka_vals))         if wbar_cka_vals else 0,
        basis_pearson_mean = float(np.mean(np.abs(basis_pr_vals))) if basis_pr_vals else 0,
        basis_cka_mean     = float(np.mean(basis_cka_vals))        if basis_cka_vals else 0,
        wbar_cka_all       = wbar_cka_vals,
        basis_cka_all      = basis_cka_vals,
    )

    print(f"\n  Off-diagonal means:")
    print(f"    W̄   |Pearson|={summary['wbar_pearson_mean']:.3f}  "
          f"CKA={summary['wbar_cka_mean']:.3f}")
    print(f"    Basis |Pearson|={summary['basis_pearson_mean']:.3f}  "
          f"CKA={summary['basis_cka_mean']:.3f}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Interpretation
# ══════════════════════════════════════════════════════════════════════════════

def interpret(label_pair: str,
              exp_A: dict,
              exp_B_A: dict,
              exp_B_B: dict):
    """
    Compare inter-model (A) vs intra-model cross-group (B) similarity.
    """
    print(f"\n{'='*70}")
    print(f"INTERPRETATION: {label_pair}")
    print(f"{'='*70}")

    if not exp_A:
        print("  No shared groups found — cannot compare A vs B.")
        return

    # Aggregate Exp A
    wbar_cka_A  = np.mean([v["wbar_cka"]  for v in exp_A.values()])
    basis_cka_A = np.mean([v["basis_cka"] for v in exp_A.values()])
    wbar_pr_A   = np.mean([abs(v["wbar_pearson"])  for v in exp_A.values()])
    basis_pr_A  = np.mean([abs(v["basis_pearson"]) for v in exp_A.values()])

    # Aggregate Exp B (average of the two models)
    wbar_cka_B  = (exp_B_A["wbar_cka_mean"]  + exp_B_B["wbar_cka_mean"])  / 2
    basis_cka_B = (exp_B_A["basis_cka_mean"] + exp_B_B["basis_cka_mean"]) / 2

    print(f"\n  Metric          | Exp A (cross-model) | Exp B (cross-group) | Winner")
    print(f"  {'-'*65}")
    def row(name, a, b):
        winner = "A>B ←" if a > b + 0.05 else ("B>A ←" if b > a + 0.05 else "≈ tie")
        print(f"  {name:<16}| {a:^19.3f} | {b:^19.3f} | {winner}")
    row("W̄ CKA",      wbar_cka_A,  wbar_cka_B)
    row("Basis CKA",  basis_cka_A, basis_cka_B)
    row("W̄ |Pearson|",wbar_pr_A,   exp_B_A["wbar_pearson_mean"])
    row("Basis |Pear|",basis_pr_A, exp_B_A["basis_pearson_mean"])

    print(f"\n  Reading:")
    # W̄
    if wbar_cka_A > wbar_cka_B + 0.1:
        print("  W̄: Same group across models is MORE similar than different groups")
        print("     within model → W̄ encodes group-specific 'role', stable across training.")
    elif wbar_cka_B > wbar_cka_A + 0.1:
        print("  W̄: Groups within model are MORE similar to each other than")
        print("     same group across models → W̄ reflects model-level topology,")
        print("     not fixed group identity.")
    else:
        print("  W̄: Cross-model and cross-group similarity are comparable.")
        print("     → W̄ is neither strongly group-specific nor model-specific.")

    # Basis
    if basis_cka_A > basis_cka_B + 0.1:
        print("  Basis: Same group across models shares address space")
        print("     → Basis directions are group-role-specific and training-invariant.")
    elif basis_cka_B > basis_cka_A + 0.1:
        print("  Basis: Groups within a model share address space more than")
        print("     same group across models → universal within-model variation modes.")
    else:
        print("  Basis: CKA is similar in both directions.")
        print("     → Basis collapse to universal subspace regardless of comparison axis.")

    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def visualize_AB(label_pair: str, exp_A: dict,
                 exp_B_A: dict, exp_B_B: dict,
                 save_path: str):
    if not exp_A:
        return

    shapes  = list(exp_A.keys())
    x       = np.arange(len(shapes))
    labels  = [str(s) for s in shapes]

    wbar_cka_A  = [exp_A[s]["wbar_cka"]   for s in shapes]
    basis_cka_A = [exp_A[s]["basis_cka"]  for s in shapes]
    wbar_pr_A   = [abs(exp_A[s]["wbar_pearson"])  for s in shapes]
    basis_pr_A  = [abs(exp_A[s]["basis_pearson"]) for s in shapes]

    # Cross-group baselines (scalars → horizontal lines)
    wbar_cka_B  = (exp_B_A["wbar_cka_mean"]  + exp_B_B["wbar_cka_mean"])  / 2
    basis_cka_B = (exp_B_A["basis_cka_mean"] + exp_B_B["basis_cka_mean"]) / 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Inter-Model vs Cross-Group Similarity\n{label_pair}",
                 fontsize=13, fontweight="bold")

    def bar_plot(ax, vals_A, baseline_B, title, ylabel):
        bars = ax.bar(x, vals_A, color="#4472C4", alpha=0.8, label="Exp A (cross-model)")
        ax.axhline(baseline_B, color="#ED7D31", linewidth=2,
                   linestyle="--", label=f"Exp B baseline (cross-group) = {baseline_B:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals_A):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    bar_plot(axes[0,0], wbar_cka_A,  wbar_cka_B,
             "W̄  CKA: same group, cross-model  vs  cross-group baseline",
             "CKA similarity")
    bar_plot(axes[0,1], basis_cka_A, basis_cka_B,
             "Basis CKA: same group, cross-model  vs  cross-group baseline",
             "CKA similarity")
    bar_plot(axes[1,0], wbar_pr_A,   exp_B_A["wbar_pearson_mean"],
             "W̄  |Pearson r|: cross-model  vs  cross-group baseline",
             "|Pearson r|")
    bar_plot(axes[1,1], basis_pr_A,  exp_B_A["basis_pearson_mean"],
             "Basis |Pearson r|: cross-model  vs  cross-group baseline",
             "|Pearson r|")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Model loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_resnet18_pretrained():
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

def load_resnet18_random():
    from torchvision.models import resnet18
    return resnet18(weights=None)

def load_gpt2():
    from transformers import AutoModel
    return AutoModel.from_pretrained("gpt2")

def load_gpt2_random():
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained("gpt2")
    return AutoModel.from_config(cfg)   # random weights, same architecture

def load_distilbert():
    from transformers import AutoModel
    return AutoModel.from_pretrained("distilbert-base-uncased")


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pairs_to_run = []

    # ── Pair 1: ResNet-18 pretrained vs ResNet-18 random ─────────────────────
    try:
        print("\n" + "█"*70)
        print("LOADING: ResNet-18 pretrained  +  ResNet-18 random init")
        print("█"*70)
        rn_pre = load_resnet18_pretrained()
        rn_rnd = load_resnet18_random()
        g_pre  = group_weights(rn_pre)
        g_rnd  = group_weights(rn_rnd)
        pairs_to_run.append((
            "ResNet-18 pretrained", g_pre,
            "ResNet-18 random",    g_rnd,
            "resnet18_pre_vs_rnd"
        ))
        print("OK")
    except Exception as e:
        print(f"ResNet-18 pair unavailable: {e}")

    # ── Pair 2: GPT-2 pretrained vs GPT-2 random ─────────────────────────────
    try:
        print("\n" + "█"*70)
        print("LOADING: GPT-2 pretrained  +  GPT-2 random init")
        print("█"*70)
        gpt_pre = load_gpt2()
        gpt_rnd = load_gpt2_random()
        g_gpre  = group_weights(gpt_pre)
        g_grnd  = group_weights(gpt_rnd)
        pairs_to_run.append((
            "GPT-2 pretrained", g_gpre,
            "GPT-2 random",    g_grnd,
            "gpt2_pre_vs_rnd"
        ))
        print("OK")
    except Exception as e:
        print(f"GPT-2 pair unavailable: {e}")

    # ── Pair 3: GPT-2 vs DistilBERT (different architecture, overlapping shapes)
    try:
        print("\n" + "█"*70)
        print("LOADING: GPT-2  +  DistilBERT  (cross-architecture)")
        print("█"*70)
        gpt_m   = load_gpt2()
        dbert   = load_distilbert()
        g_gpt   = group_weights(gpt_m)
        g_db    = group_weights(dbert)
        shared  = set(g_gpt) & set(g_db)
        if shared:
            pairs_to_run.append((
                "GPT-2", g_gpt,
                "DistilBERT", g_db,
                "gpt2_vs_distilbert"
            ))
            print(f"OK — {len(shared)} shared shapes: {shared}")
        else:
            print("No overlapping group shapes between GPT-2 and DistilBERT.")
    except Exception as e:
        print(f"Cross-arch pair unavailable: {e}")

    if not pairs_to_run:
        print("No model pairs loaded. Exiting.")
        return

    # ── Run each pair ─────────────────────────────────────────────────────────
    for label_A, groups_A, label_B, groups_B, suffix in pairs_to_run:
        print(f"\n\n{'#'*70}")
        print(f"# PAIR: {label_A}  vs  {label_B}")
        print(f"{'#'*70}")

        # Exp A
        res_A = experiment_A(label_A, groups_A, label_B, groups_B)

        # Exp B — cross-group baseline for each model separately
        res_B_A = experiment_B(label_A, groups_A)
        res_B_B = experiment_B(label_B, groups_B)

        # Interpret
        interpret(f"{label_A} vs {label_B}", res_A, res_B_A, res_B_B)

        # Visualize
        save_path = os.path.join(script_dir, f"intermodel_{suffix}.png")
        visualize_AB(f"{label_A} vs {label_B}", res_A, res_B_A, res_B_B, save_path)

    print("\n\nAll pairs processed.")


if __name__ == "__main__":
    run()
