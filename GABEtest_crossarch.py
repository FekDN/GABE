# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_crossarch.py — Experiment 22: Cross-Architecture Test
#
# PURPOSE:
#   Tests whether the GABE spectral elevation generalizes across
#   fundamentally different neural network architectures:
#     - ResNet-18  (skip connections, BatchNorm)
#     - VGG-11     (sequential conv, no skip connections)
#     - MobileNetV2 (depthwise separable convolutions)
#
#   If the effect is architecture-generic, all three should show
#   similar spectral elevation for their respective Conv2d groups.
#
#   Also tests whether basis subspaces of matching shapes ACROSS
#   architectures show alignment (CKA test, analogous to Experiment 6).
#
# USAGE:
#   python GABEtest_crossarch.py
#   python GABEtest_crossarch.py --n_grad 64 --n_spectrum 300

import sys, os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from collections import defaultdict
from scipy.stats import percentileofscore
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_conv_layers(model):
    """Returns {shape: [weight_tensors]} for all multi-layer Conv2d groups."""
    groups = defaultdict(list)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            groups[tuple(m.weight.shape)].append(m.weight.detach())
    return {k: v for k, v in groups.items() if len(v) >= 2}


def extract_basis(weights):
    gabe = GABE()
    _, B_s, _, _ = gabe._extract_svd_components(weights)
    K = B_s.shape[0];  D = B_s[0].numel()
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    return Q[:, :K], D


def subspace_alignment(B1, B2):
    _, S, _ = torch.linalg.svd(B1.T @ B2)
    return (S ** 2).mean().item()


def build_fisher_mvp(model, param, loader, loss_fn, device, n_grad):
    model.eval()
    grads, count = [], 0
    for xb, yb in loader:
        for i in range(xb.size(0)):
            if count >= n_grad: break
            x, y = xb[i:i+1].to(device), yb[i:i+1].to(device)
            model.zero_grad(); loss_fn(model(x), y).backward()
            if param.grad is not None:
                grads.append(param.grad.detach().reshape(-1).clone())
                param.grad = None
            count += 1
        if count >= n_grad: break
    if not grads: return None, 0.0
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    return fvp, (G**2).sum(1).mean().item()


def spectral_percentile(B, fvp, D, n_samples, device):
    rq_r = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_r.append((v @ fvp(v)).item())
    rq_r = np.array(rq_r)
    K = B.shape[1]
    rq_g = np.array([(B[:, k] @ fvp(B[:, k])).item() for k in range(K)])
    pcts = np.array([percentileofscore(rq_r, r) for r in rq_g])
    return pcts, rq_g.mean() / (rq_r.mean() + 1e-12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_grad=64, n_spectrum=300, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 22: Cross-Architecture Test")
    print("=" * 62)

    import torchvision.transforms as transforms
    tf = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, list(range(n_grad))),
        batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    models = {
        "ResNet-18":    torchvision.models.resnet18(weights="IMAGENET1K_V1"),
        "VGG-11":       torchvision.models.vgg11(weights="IMAGENET1K_V1"),
        "MobileNetV2":  torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1"),
    }

    arch_results = {}   # {arch: {shape: {K, D, mean_pct, ratio, B}}}

    for arch_name, model in models.items():
        print(f"\n[{arch_name}] analyzing...")
        model = model.eval().to(device)
        groups = group_conv_layers(model)
        arch_results[arch_name] = {}

        for shape, ws in sorted(groups.items()):
            # Find corresponding parameter
            param = next(
                (m.weight for m in model.modules()
                 if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == shape),
                None)
            if param is None: continue

            fvp, trace_F = build_fisher_mvp(model, param, loader, loss_fn, device, n_grad)
            if fvp is None or trace_F < 1e-12: continue

            B, D = extract_basis(ws)
            K = B.shape[1]
            pcts, ratio = spectral_percentile(B, fvp, D, n_spectrum, device)
            arch_results[arch_name][shape] = dict(K=K, D=D, L=len(ws),
                                                   mean_pct=pcts.mean(),
                                                   ratio=ratio, B=B, pcts=pcts)
            print(f"  {str(shape):<24} L={len(ws):>2}  K={K}  D={D:>7}  "
                  f"pct={pcts.mean():5.1f}th  ratio={ratio:5.2f}×")

    # --- Per-architecture summary ---
    print()
    print("=" * 62)
    print("ARCHITECTURE COMPARISON — mean spectral percentile per arch")
    print("=" * 62)
    arch_means = {}
    for arch_name, shape_data in arch_results.items():
        if not shape_data: continue
        all_pcts = np.concatenate([r['pcts'] for r in shape_data.values()])
        mean_pct = all_pcts.mean()
        arch_means[arch_name] = mean_pct
        bar = "█" * int(mean_pct / 5)
        print(f"  {arch_name:<16}  {mean_pct:5.1f}th  {bar}")

    # --- Cross-architecture subspace alignment for matching shapes ---
    print()
    print("Cross-architecture subspace alignment (matching shapes):")
    print(f"  {'Shape':<24} {'Arch A':<14} {'Arch B':<14} {'Alignment':>10}")
    print("  " + "-" * 66)
    arch_names = list(arch_results.keys())
    for a, b in combinations(arch_names, 2):
        common_shapes = set(arch_results[a].keys()) & set(arch_results[b].keys())
        for shape in sorted(common_shapes):
            B_a = arch_results[a][shape]['B']
            B_b = arch_results[b][shape]['B']
            # Align to same K
            K = min(B_a.shape[1], B_b.shape[1])
            sa = subspace_alignment(B_a[:, :K], B_b[:, :K])
            D = B_a.shape[0]; rand_expected = K / D
            print(f"  {str(shape):<24} {a:<14} {b:<14} {sa:>10.6f}  "
                  f"(rand={rand_expected:.6f})")

    print()
    min_pct = min(arch_means.values()) if arch_means else 0
    max_pct = max(arch_means.values()) if arch_means else 0
    spread  = max_pct - min_pct

    if min_pct > 65:
        conclusion = "UNIVERSAL — spectral elevation present in all tested architectures."
    elif min_pct > 50 and spread < 20:
        conclusion = "MOSTLY CONSISTENT — mild variation, all architectures above random."
    else:
        conclusion = f"ARCHITECTURE-DEPENDENT — range {min_pct:.0f}–{max_pct:.0f}th. Effect not universal."
    print(f"Conclusion: {conclusion}")
    return arch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grad",     type=int, default=64)
    parser.add_argument("--n_spectrum", type=int, default=300)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run(args.n_grad, args.n_spectrum, args.device, args.seed)
