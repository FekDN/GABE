# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_width.py — Experiment 15: Width Sweep
#
# PURPOSE:
#   Tests whether the spectral elevation of GABE directions persists
#   as the weight dimension D grows. As D increases:
#   - K/D (random baseline) shrinks — harder to beat by chance
#   - If elevation is real, it should scale proportionally with D
#   - If it's a small-D artifact, it should disappear at large widths
#
#   Uses SmallConvNet trained with varying channel widths C ∈ {16, 32, 64, 128},
#   producing GABE groups of shape [C, C, 3, 3] → D = C² × 9.
#
# METHOD:
#   For each width C:
#     1. Train SmallConvNet(C) for a fixed number of epochs.
#     2. Extract GABE basis (K=3, always L=4 layers).
#     3. Build empirical Fisher MVP from training data.
#     4. Report spectral percentile and Rayleigh ratio.
#
# KEY PREDICTION:
#   If the effect is real:    percentile ≈ constant across widths (structure scales with D).
#   If it is a small-D artifact: percentile → ~50th as D grows.
#
# USAGE:
#   python GABEtest_width.py
#   python GABEtest_width.py --widths 16 32 64 128 --epochs 20

import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from scipy.stats import percentileofscore

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


class SmallConvNet(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.stem = nn.Conv2d(3, C, 3, padding=1)
        self.c1 = nn.Conv2d(C, C, 3, padding=1)
        self.c2 = nn.Conv2d(C, C, 3, padding=1)
        self.c3 = nn.Conv2d(C, C, 3, padding=1)
        self.c4 = nn.Conv2d(C, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Linear(C * 16, 10)
        self._C = C

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for l in [self.c1, self.c2, self.c3, self.c4]:
            x = torch.relu(l(x))
        return self.head(self.pool(x).flatten(1))


def train_model(C, epochs, n_samples, device, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf),
            list(range(n_samples))),
        batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    model = SmallConvNet(C).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model.eval()


def build_fisher_mvp(model, param, loader, loss_fn, device, n_grad):
    model.eval()
    grads, count = [], 0
    for xb, yb in loader:
        for i in range(xb.size(0)):
            if count >= n_grad: break
            x, y = xb[i:i+1].to(device), yb[i:i+1].to(device)
            model.zero_grad(); loss_fn(model(x), y).backward()
            grads.append(param.grad.detach().reshape(-1).clone())
            param.grad = None; count += 1
        if count >= n_grad: break
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    return fvp, (G**2).sum(1).mean().item()


def percentile_analysis(B, mvp, D, n_samples=500, device="cpu"):
    rq_r = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_r.append((v @ mvp(v)).item())
    rq_r = np.array(rq_r)
    K = B.shape[1]
    rq_g = np.array([(B[:, k] @ mvp(B[:, k])).item() for k in range(K)])
    pcts = np.array([percentileofscore(rq_r, r) for r in rq_g])
    return rq_g, pcts, rq_r


def run(widths=(16, 32, 64, 128), epochs=20, n_samples=2000,
        n_grad=64, n_spectrum=500, device="cpu", seed=42):

    print("=" * 62)
    print("GABE Experiment 15: Width Sweep")
    print("=" * 62)
    print(f"widths={widths}  epochs={epochs}  n_samples={n_samples}")
    print()

    loss_fn = nn.CrossEntropyLoss()
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    grad_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf),
            list(range(n_grad))),
        batch_size=1, shuffle=False)

    print(f"{'C':>6} {'D':>8} {'K':>4} {'K/D (rand)':>12} {'mean_pct':>10} "
          f"{'rq_ratio':>10} {'p_above99':>12}")
    print("-" * 68)

    rows = []
    for C in widths:
        print(f"  C={C}: training...", end=" ", flush=True)
        model = train_model(C, epochs, n_samples, device, seed)
        gabe  = GABE()
        ws = [model.c1.weight.detach(), model.c2.weight.detach(),
              model.c3.weight.detach(), model.c4.weight.detach()]
        _, B_s, _, _ = gabe._extract_svd_components(ws)
        K = B_s.shape[0];  D = B_s[0].numel()
        B_flat = B_s.view(K, D).T.float()
        Q, _ = torch.linalg.qr(B_flat);  B_ortho = Q[:, :K]

        print("fisher...", end=" ", flush=True)
        fvp, _ = build_fisher_mvp(model, model.c1.weight,
                                   grad_loader, loss_fn, device, n_grad)
        rq_g, pcts, rq_r = percentile_analysis(B_ortho, fvp, D, n_spectrum, device)
        ratio = rq_g.mean() / (rq_r.mean() + 1e-12)
        p_99  = (pcts >= 99).mean()

        print(f"done.  pct={pcts.mean():.1f}th  ratio={ratio:.2f}x")
        print(f"  {C:>6} {D:>8} {K:>4} {K/D:>12.6f} {pcts.mean():>10.1f} "
              f"{ratio:>10.2f}× {p_99:>12.2f}")
        rows.append(dict(C=C, D=D, K=K, mean_pct=pcts.mean(), ratio=ratio,
                         pcts=pcts, rq_g=rq_g))

    print()
    print("=" * 62)
    print("SCALING ANALYSIS")
    print("=" * 62)
    print("  C      D    K    mean_pct   rq_ratio")
    for r in rows:
        bar = "█" * int(r['mean_pct'] / 5)
        print(f"  {r['C']:>4}  {r['D']:>6}  {r['K']:>2}   "
              f"{r['mean_pct']:>7.1f}th  {r['ratio']:>8.2f}×   {bar}")

    first_pct = rows[0]['mean_pct']
    last_pct  = rows[-1]['mean_pct']
    drop = first_pct - last_pct

    print()
    if drop < 10 and last_pct > 60:
        conclusion = "ROBUST TO WIDTH — spectral elevation persists as D grows."
    elif drop > 20 and last_pct < 55:
        conclusion = "WIDTH-SENSITIVE — effect weakens at large D. Possible small-D artifact."
    else:
        conclusion = f"MIXED — {first_pct:.0f}th → {last_pct:.0f}th over width range."
    print(f"Conclusion: {conclusion}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths",     type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--n_samples",  type=int, default=2000)
    parser.add_argument("--n_grad",     type=int, default=64)
    parser.add_argument("--n_spectrum", type=int, default=500)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.widths), args.epochs, args.n_samples,
        args.n_grad, args.n_spectrum, args.device, args.seed)
