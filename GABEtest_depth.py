# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_depth.py — Experiment 14: Scale Depth Sweep
#
# PURPOSE:
#   Tests how GABE properties scale with the number of layers L in a group:
#   - K = L-1 by construction; does the basis quality scale with K?
#   - Does the Rayleigh percentile stay elevated as L grows?
#   - Is there a phase transition depth where structure emerges?
#
#   Uses pretrained ResNet family (ResNet-18/34/50/101) which differ in depth
#   while sharing the same Conv2d shapes. Also tests within a single model
#   by varying how many layers are included in the GABE group.
#
# METHOD:
#   For each depth D in [2, 4, 8, L_max]:
#     1. Take the first D layers from a fixed group of same-shaped Conv2d layers.
#     2. Extract GABE basis (K = D-1).
#     3. Compute Rayleigh quotients using empirical Fisher MVP.
#     4. Report spectral percentile.
#
# USAGE:
#   python GABEtest_depth.py
#   python GABEtest_depth.py --model resnet18 --shape 64 64 3 3 --n_grad 64

import sys, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from scipy.stats import percentileofscore

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str) -> nn.Module:
    if model_name == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    elif model_name == "resnet34":
        m = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    elif model_name == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m.eval().to(device)


def get_group_layers(model: nn.Module, target_shape: tuple) -> list:
    """Returns all Conv2d weight tensors with given shape, sorted by module order."""
    ws = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == target_shape:
            ws.append(m.weight.detach())
    return ws


def build_fisher_mvp(model, loss_fn, loader, param, device, n_grad):
    model.eval()
    grads = []
    count = 0
    for xb, yb in loader:
        for i in range(xb.size(0)):
            if count >= n_grad: break
            x, y = xb[i:i+1].to(device), yb[i:i+1].to(device)
            model.zero_grad()
            loss_fn(model(x), y).backward()
            grads.append(param.grad.detach().reshape(-1).clone())
            param.grad = None
            count += 1
        if count >= n_grad: break
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    trace = (G ** 2).sum(1).mean().item()
    return fvp, trace


def spectral_percentile(B_gabe, mvp, D, n_samples=500, device="cpu"):
    rq_rand = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_rand.append((v @ mvp(v)).item())
    rq_rand = np.array(rq_rand)
    K = B_gabe.shape[1]
    rq_gabe = np.array([(B_gabe[:, k] @ mvp(B_gabe[:, k])).item() for k in range(K)])
    pcts = np.array([percentileofscore(rq_rand, rq) for rq in rq_gabe])
    return rq_gabe, pcts, rq_rand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model_name: str = "resnet18",
    target_shape: tuple = (64, 64, 3, 3),
    n_grad: int = 64,
    n_spectrum: int = 500,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed); np.random.seed(seed)
    print("=" * 62)
    print("GABE Experiment 14: Depth Sweep")
    print("=" * 62)
    print(f"Model={model_name}  shape={target_shape}  n_grad={n_grad}")
    print()

    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, list(range(n_grad))),
        batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[1] Loading {model_name}...")
    model = load_model(model_name, device)

    all_layers = get_group_layers(model, target_shape)
    L_max = len(all_layers)
    print(f"  Found {L_max} layers with shape {target_shape}")

    if L_max < 2:
        print("  ERROR: need ≥2 layers. Try a different shape or model.")
        return

    # Get parameter for Fisher MVP (first matching layer)
    param = next(m.weight for m in model.modules()
                 if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == target_shape)
    D_single = param.numel()

    print(f"[2] Building Fisher MVP (n_grad={n_grad})...")
    fvp, trace_F = build_fisher_mvp(model, loss_fn, loader, param, device, n_grad)

    # Depth sweep: 2, 4, max (and powers of 2 in between)
    depths = sorted(set([2, 4, 8, L_max]))
    depths = [d for d in depths if d <= L_max]

    print(f"\n[3] Depth sweep: {depths}")
    print()
    print(f"  {'L':>4} {'K':>4} {'mean_pct':>10} {'min_pct':>10} {'max_pct':>10} "
          f"{'rq_mean':>10} {'rq/rq_rand':>12}")
    print("  " + "-" * 66)

    results = []
    for L in depths:
        ws = all_layers[:L]
        gabe = GABE()
        _, B_stacked, _, _ = gabe._extract_svd_components(ws)
        K = B_stacked.shape[0]
        B_flat = B_stacked.view(K, D_single).T.float()
        Q, _ = torch.linalg.qr(B_flat);  B_ortho = Q[:, :K]

        rq_gabe, pcts, rq_rand = spectral_percentile(B_ortho, fvp, D_single,
                                                      n_spectrum, device)
        rand_mean = rq_rand.mean()
        ratio = rq_gabe.mean() / (rand_mean + 1e-12)
        results.append(dict(L=L, K=K, pcts=pcts, rq_gabe=rq_gabe,
                            ratio=ratio, rand_mean=rand_mean))
        print(f"  {L:>4} {K:>4} {pcts.mean():>10.1f} {pcts.min():>10.1f} "
              f"{pcts.max():>10.1f} {rq_gabe.mean():>10.6f} {ratio:>12.2f}×")

    print()
    print("=" * 62)
    print("INTERPRETATION")
    print("=" * 62)
    pcts_by_depth = [(r['L'], r['pcts'].mean()) for r in results]
    print("  L  →  mean_percentile")
    for L, pct in pcts_by_depth:
        bar = "█" * int(pct / 5)
        print(f"  {L:>3}  →  {pct:5.1f}th  {bar}")

    first_pct = pcts_by_depth[0][1]
    last_pct  = pcts_by_depth[-1][1]
    if last_pct >= first_pct - 5:
        trend = "STABLE — percentile does not degrade with depth."
    elif last_pct < 50:
        trend = "DEGRADES — percentile drops toward random as L grows."
    else:
        trend = f"MIXED — starts at {first_pct:.0f}th, ends at {last_pct:.0f}th."
    print(f"\nTrend: {trend}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="resnet18")
    parser.add_argument("--shape",    type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--n_grad",   type=int, default=64)
    parser.add_argument("--n_spectrum", type=int, default=500)
    parser.add_argument("--device",   type=str, default="cpu")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    run(args.model, tuple(args.shape), args.n_grad, args.n_spectrum,
        args.device, args.seed)
