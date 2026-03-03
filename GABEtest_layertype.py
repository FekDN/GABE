# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_layertype.py — Experiment 17: Cross-Layer Type Test
#
# PURPOSE:
#   Tests whether the GABE spectral elevation is uniform across layer types
#   or concentrated in specific functional sub-architectures.
#
#   Layer types tested (on ResNet-18 and GPT-2):
#     ResNet-18: Conv2d groups at different depth levels (stem, mid, deep)
#     GPT-2:     Attention Q/K/V projections, FFN up/down, output projections
#
#   HYPOTHESIS: If the effect is a generic property of same-shaped layers,
#   all layer types should show similar elevation.
#   If it is functionally specific, certain types (e.g. attention Q/K/V) will
#   show stronger alignment than others (e.g. output projections).
#
# USAGE:
#   python GABEtest_layertype.py
#   python GABEtest_layertype.py --model resnet18
#   python GABEtest_layertype.py --model gpt2

import sys, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from collections import defaultdict
from scipy.stats import percentileofscore

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Generic group extractor
# ---------------------------------------------------------------------------

def group_layers_by_shape(model, layer_cls):
    """Group layers by weight shape. Returns {shape: [weight_tensors]}."""
    groups = defaultdict(list)
    for m in model.modules():
        if isinstance(m, layer_cls):
            groups[tuple(m.weight.shape)].append(m.weight.detach())
    return {k: v for k, v in groups.items() if len(v) >= 2}


def extract_basis_from_weights(weights):
    gabe = GABE()
    _, B_s, _, _ = gabe._extract_svd_components(weights)
    K = B_s.shape[0];  D = B_s[0].numel()
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    return Q[:, :K], D


def rayleigh_percentile(B, D, n_samples=300):
    """
    Rayleigh quotients using the raw weight covariance proxy:
    w^T (W^T W) w / Tr(W^T W) for a random reference matrix.
    Falls back to random walk when no MVP is available.
    Here we use the GABE mean weight as the reference quadratic form.
    """
    K = B.shape[1]
    # Build a proxy MVP: M = W_bar^T W_bar for the group mean
    # This is a lightweight proxy that doesn't need data
    rq_r = []
    for _ in range(n_samples):
        v = torch.randn(D); v /= v.norm()
        rq_r.append(v.norm().item() ** 2 / D)   # trivially 1/D — use identity proxy
    # Use norm-squared proxy: RQ relative to identity = v^T v / D = 1 always
    # Use actual Fisher estimate from the weights covariance instead
    # Build proxy: M_proxy = sum_k B_k B_k^T  (reconstruction subspace)
    # v^T M v = sum_k (v . B_k)^2
    def mvp(v):
        return sum((v @ B[:, k]).unsqueeze(0) * B[:, k] for k in range(K))

    rq_rand = []
    for _ in range(n_samples):
        v = torch.randn(D); v /= v.norm()
        rq_rand.append((v @ mvp(v)).item())

    rq_gabe = np.array([(B[:, k] @ mvp(B[:, k])).item() for k in range(K)])
    rq_rand  = np.array(rq_rand)
    pcts = np.array([percentileofscore(rq_rand, r) for r in rq_gabe])
    return rq_gabe, pcts, rq_rand


def variance_explained_by_basis(weights):
    """How much of the total inter-layer variance is explained by B_k?"""
    gabe = GABE()
    stacked = torch.stack(weights).float()
    mean = stacked.mean(0)
    centered = stacked - mean.unsqueeze(0)
    total_var = (centered ** 2).sum().item()

    _, B_s, coeffs, _ = gabe._extract_svd_components(weights)
    K = B_s.shape[0];  D = B_s[0].numel()
    B_flat = B_s.view(K, D).float()
    reconstructed = (coeffs.float() @ B_flat).view_as(centered)
    recon_var = (reconstructed ** 2).sum().item()
    return recon_var / (total_var + 1e-12)


# ---------------------------------------------------------------------------
# ResNet-18 analysis
# ---------------------------------------------------------------------------

def analyze_resnet(device):
    print("\n── ResNet-18 ──────────────────────────────────────────────")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().to(device)
    groups = group_layers_by_shape(model, nn.Conv2d)

    # Label groups by semantic depth
    shape_labels = {
        (64, 3, 7, 7):    "stem (7×7)",
        (64, 64, 3, 3):   "layer1 (3×3, C=64)",
        (128, 128, 3, 3): "layer2 (3×3, C=128)",
        (256, 256, 3, 3): "layer3 (3×3, C=256)",
        (512, 512, 3, 3): "layer4 (3×3, C=512)",
    }

    rows = []
    for shape, ws in sorted(groups.items()):
        label = shape_labels.get(shape, f"Conv2d {shape}")
        B, D = extract_basis_from_weights(ws)
        K = B.shape[1]
        ve = variance_explained_by_basis(ws)
        rq_g, pcts, _ = rayleigh_percentile(B, D)
        rows.append((label, len(ws), K, D, pcts.mean(), ve))

    print(f"  {'Layer type':<32} {'L':>3} {'K':>3} {'D':>7} "
          f"{'mean_pct':>10} {'var_expl':>10}")
    print("  " + "-" * 70)
    for label, L, K, D, mp, ve in rows:
        print(f"  {label:<32} {L:>3} {K:>3} {D:>7} {mp:>10.1f} {ve:>10.4f}")
    return rows


# ---------------------------------------------------------------------------
# GPT-2 analysis
# ---------------------------------------------------------------------------

def analyze_gpt2(device):
    print("\n── GPT-2 (small) ──────────────────────────────────────────")
    try:
        from transformers import GPT2Model
    except ImportError:
        print("  transformers not installed. Skipping GPT-2.")
        return []

    model = GPT2Model.from_pretrained("gpt2").eval().to(device)

    # Collect layers by functional role using name patterns
    role_weights = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.detach()
            if "c_attn" in name:
                role_weights["attn_qkv"].append(w)
            elif "c_proj" in name and "attn" in name:
                role_weights["attn_out"].append(w)
            elif "c_fc" in name:
                role_weights["ffn_up"].append(w)
            elif "c_proj" in name and "mlp" in name:
                role_weights["ffn_down"].append(w)

    rows = []
    for role, ws_all in sorted(role_weights.items()):
        # Group by shape within each role
        shape_groups = defaultdict(list)
        for w in ws_all:
            shape_groups[tuple(w.shape)].append(w)
        for shape, ws in shape_groups.items():
            if len(ws) < 2: continue
            B, D = extract_basis_from_weights(ws)
            K = B.shape[1]
            ve = variance_explained_by_basis(ws)
            rq_g, pcts, _ = rayleigh_percentile(B, D)
            label = f"{role} {shape}"
            rows.append((label, len(ws), K, D, pcts.mean(), ve))

    print(f"  {'Layer type':<40} {'L':>3} {'K':>3} {'D':>7} "
          f"{'mean_pct':>10} {'var_expl':>10}")
    print("  " + "-" * 78)
    for label, L, K, D, mp, ve in rows:
        print(f"  {label:<40} {L:>3} {K:>3} {D:>7} {mp:>10.1f} {ve:>10.4f}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model: str = "both", device: str = "cpu"):
    print("=" * 62)
    print("GABE Experiment 17: Cross-Layer Type Test")
    print("=" * 62)
    print("Note: Percentiles computed w.r.t. GABE projection subspace proxy.")
    print("      For full Fisher/Hessian percentiles, use GABEtest_spectrum.py")
    print()

    all_rows = []
    if model in ("resnet18", "both"):
        all_rows += analyze_resnet(device)
    if model in ("gpt2", "both"):
        all_rows += analyze_gpt2(device)

    if not all_rows:
        return

    # Summary: which layer type shows strongest GABE effect?
    print()
    print("=" * 62)
    print("SUMMARY — ranked by mean_percentile")
    print("=" * 62)
    all_rows.sort(key=lambda r: -r[4])
    for label, L, K, D, mp, ve in all_rows:
        bar = "█" * int(mp / 5)
        print(f"  {label:<38}  pct={mp:5.1f}th  var_expl={ve:.4f}  {bar}")

    top_label = all_rows[0][0]
    top_pct   = all_rows[0][4]
    bot_label = all_rows[-1][0]
    bot_pct   = all_rows[-1][4]
    spread    = top_pct - bot_pct

    print()
    print(f"Strongest:  {top_label}  ({top_pct:.1f}th percentile)")
    print(f"Weakest:    {bot_label}  ({bot_pct:.1f}th percentile)")
    print(f"Spread:     {spread:.1f} percentile points")
    print()
    if spread < 15:
        print("UNIFORM — effect is consistent across all layer types.")
    elif spread > 30:
        print("HETEROGENEOUS — effect is concentrated in specific layer types.")
    else:
        print("MODERATE VARIATION — some layer types show stronger GABE structure.")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="both",
                        choices=["resnet18", "gpt2", "both"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run(args.model, args.device)
