# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_init.py — Experiment 16: Random Reinitialization Control
#
# PURPOSE:
#   Determines *when* the functional structure in the GABE basis emerges
#   during training. Tracks spectral percentile of GABE directions at:
#     - Random initialization (epoch 0)
#     - After 1, 5, 10, 20, full epochs
#
#   CRITICAL QUESTION: Does the elevated percentile exist at initialization,
#   or does it emerge during training?
#
#   Expected outcomes:
#     A. Percentile ≈ 50 at init, rises during training
#        → structure is learned, not inherent to SVD on same-shaped matrices
#     B. Percentile already elevated at init
#        → may be an SVD artifact independent of training
#     C. Non-monotone: rises then falls (e.g. grokking-type transition)
#        → interesting training dynamics
#
# USAGE:
#   python GABEtest_init.py
#   python GABEtest_init.py --epochs 30 --checkpoints 0 1 5 10 20 30

import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from scipy.stats import percentileofscore
import copy

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

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for l in [self.c1, self.c2, self.c3, self.c4]:
            x = torch.relu(l(x))
        return self.head(self.pool(x).flatten(1))

    def gabe_weights(self):
        return [self.c1.weight.detach(), self.c2.weight.detach(),
                self.c3.weight.detach(), self.c4.weight.detach()]


def extract_basis(model):
    gabe = GABE()
    ws = model.gabe_weights()
    _, B_s, _, _ = gabe._extract_svd_components(ws)
    K = B_s.shape[0];  D = B_s[0].numel()
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    return Q[:, :K]


def build_fisher_mvp(model, param, grad_data, loss_fn, device):
    model.eval()
    grads = []
    for x, y in grad_data:
        x, y = x.unsqueeze(0).to(device), torch.tensor([y]).to(device)
        model.zero_grad(); loss_fn(model(x), y).backward()
        grads.append(param.grad.detach().reshape(-1).clone())
        param.grad = None
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    return fvp


def spectral_percentile(B, fvp, n_samples=300, device="cpu"):
    D = B.shape[0]
    rq_r = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_r.append((v @ fvp(v)).item())
    rq_r = np.array(rq_r)
    K = B.shape[1]
    rq_g = np.array([(B[:, k] @ fvp(B[:, k])).item() for k in range(K)])
    pcts = np.array([percentileofscore(rq_r, r) for r in rq_g])
    return rq_g, pcts


def run(epochs=20, C=32, n_samples=2000, n_grad=32,
        checkpoints=None, device="cpu", seed=42):

    if checkpoints is None:
        checkpoints = [0, 1, 3, 5, 10, epochs]
    checkpoints = sorted(set([0] + checkpoints + [epochs]))

    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 16: Reinitialization Control")
    print("=" * 62)
    print(f"epochs={epochs}  C={C}  n_samples={n_samples}  n_grad={n_grad}")
    print(f"checkpoints={checkpoints}")
    print()

    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    full_ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_ds, list(range(n_samples))),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    # Small fixed grad set for Fisher
    grad_data = [full_ds[i] for i in range(n_grad)]
    loss_fn = nn.CrossEntropyLoss()

    model = SmallConvNet(C).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    print(f"  {'epoch':>6} {'mean_pct':>10} {'max_pct':>10} {'rq_mean':>10} "
          f"{'train_acc':>10}")
    print("  " + "-" * 52)

    results = []
    current_epoch = 0

    def snapshot(ep, acc):
        B = extract_basis(model)
        fvp = build_fisher_mvp(model, model.c1.weight, grad_data, loss_fn, device)
        rq_g, pcts = spectral_percentile(B, fvp, device=device)
        print(f"  {ep:>6} {pcts.mean():>10.1f} {pcts.max():>10.1f} "
              f"{rq_g.mean():>10.6f} {acc:>10.3f}")
        results.append(dict(epoch=ep, pcts=pcts, rq_g=rq_g, acc=acc))

    # epoch 0: random init
    snapshot(0, 0.0)

    for ep in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
        acc = correct / total
        model.eval()
        if ep in checkpoints:
            snapshot(ep, acc)

    print()
    print("=" * 62)
    print("EMERGENCE ANALYSIS")
    print("=" * 62)
    init_pct  = results[0]['pcts'].mean()
    final_pct = results[-1]['pcts'].mean()
    peak_pct  = max(r['pcts'].mean() for r in results)
    peak_ep   = results[[r['pcts'].mean() for r in results].index(peak_pct)]['epoch']

    print(f"  init  percentile : {init_pct:.1f}th")
    print(f"  final percentile : {final_pct:.1f}th")
    print(f"  peak  percentile : {peak_pct:.1f}th  (epoch {peak_ep})")
    print(f"  gain over training: {final_pct - init_pct:+.1f}")
    print()

    if init_pct > 70 and final_pct > 70:
        verdict = "PRE-EXISTING — structure present at init; may be SVD artifact."
    elif init_pct < 55 and final_pct > 70:
        verdict = "LEARNED — structure emerges during training. Non-trivial."
    elif init_pct < 55 and final_pct < 60:
        verdict = "NOT LEARNED — percentile stays near random throughout training."
    else:
        verdict = f"PARTIAL — moderate rise from {init_pct:.0f}th to {final_pct:.0f}th."
    print(f"Verdict: {verdict}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--C",           type=int, default=32)
    parser.add_argument("--n_samples",   type=int, default=2000)
    parser.add_argument("--n_grad",      type=int, default=32)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    run(args.epochs, args.C, args.n_samples, args.n_grad,
        args.checkpoints, args.device, args.seed)
