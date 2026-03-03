# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_dynamics.py — Experiment 25: Training Dynamics Tracking
#
# PURPOSE:
#   Tracks how GABE components (W_bar, B_k, α_i) evolve during training —
#   specifically:
#     - When does B_k subspace "lock in" to its final orientation?
#     - Does the curvature alignment (spectral percentile) emerge early or late?
#     - How does ||ΔW_bar|| vs ||Δα|| scale during training?
#     - Is there a phase transition in any metric?
#
#   TRACKING METRICS (at each checkpoint):
#     (a) Subspace stability: alignment of B_k(t) with B_k(final)
#     (b) Spectral percentile of B_k(t) in Fisher CDF
#     (c) Relative norms: ||W_bar(t) - W_bar(0)|| vs ||α(t) - α(0)||
#     (d) Training accuracy
#
#   EXPECTED PATTERNS:
#     "Early lock-in": B_k subspace stabilizes in the first few epochs
#       while accuracy is still low. Suggests the structural manifold is
#       determined before the model learns the task.
#     "Co-emergence": spectral percentile rises in parallel with accuracy.
#       Suggests GABE structure is learned alongside task knowledge.
#     "Late specialization": structure emerges only after accuracy plateaus.
#       Suggests a fine-tuning phase of structural organization.
#
# USAGE:
#   python GABEtest_dynamics.py
#   python GABEtest_dynamics.py --epochs 50 --C 32 --checkpoints 0 1 2 5 10 20 50

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

    def group_weights(self):
        return [l.weight.detach().clone()
                for l in [self.c1, self.c2, self.c3, self.c4]]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def extract_basis(weights):
    gabe = GABE()
    _, B_s, coeffs, _ = gabe._extract_svd_components(weights)
    K = B_s.shape[0];  D = B_s[0].numel()
    w_bar = torch.stack(weights).float().mean(0)
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    return Q[:, :K], coeffs.detach(), w_bar, D


def subspace_alignment(B1, B2):
    K = min(B1.shape[1], B2.shape[1])
    _, S, _ = torch.linalg.svd(B1[:, :K].T @ B2[:, :K])
    return (S ** 2).mean().item()


def build_fisher_mvp(model, param, grad_data, loss_fn, device):
    model.eval()
    grads = []
    for x, y in grad_data:
        x, y = x.unsqueeze(0).to(device), torch.tensor([y]).to(device)
        model.zero_grad(); loss_fn(model(x), y).backward()
        if param.grad is not None:
            grads.append(param.grad.detach().reshape(-1).clone())
            param.grad = None
    if not grads: return None
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    return fvp


def spectral_percentile(B, fvp, D, n_samples=200, device="cpu"):
    rq_r = [0.0]
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_r.append((v @ fvp(v)).item())
    rq_r = np.array(rq_r[1:])
    rq_g = np.array([(B[:, k] @ fvp(B[:, k])).item() for k in range(B.shape[1])])
    return np.array([percentileofscore(rq_r, r) for r in rq_g])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(epochs=30, C=32, n_samples=2000, n_grad=32,
        checkpoints=None, device="cpu", seed=42):

    if checkpoints is None:
        checkpoints = [0, 1, 2, 5, 10, 20, epochs]
    checkpoints = sorted(set([0] + checkpoints + [epochs]))

    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 25: Training Dynamics Tracking")
    print("=" * 62)
    print(f"epochs={epochs}  C={C}  n_samples={n_samples}")
    print(f"checkpoints={checkpoints}")
    print()

    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    full_ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_ds, list(range(n_samples))),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    grad_data = [full_ds[i] for i in range(n_grad)]
    loss_fn = nn.CrossEntropyLoss()

    model = SmallConvNet(C).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    print(f"  {'ep':>4} {'acc':>6} {'pct':>8} {'sa_final':>10} {'dW_bar':>10} "
          f"{'dAlpha':>10}")
    print("  " + "-" * 54)

    snapshots = []
    B_final = None   # filled after training completes (two-pass approach)
    alpha_init = None
    w_bar_init = None

    def snapshot(ep, acc):
        ws = model.group_weights()
        B, coeffs, w_bar, D = extract_basis(ws)
        fvp = build_fisher_mvp(model, model.c1.weight, grad_data, loss_fn, device)
        pcts = spectral_percentile(B, fvp, D, device=device) if fvp else np.array([50.0])

        # Drift from initialization
        if alpha_init is not None:
            d_alpha = (coeffs.float() - alpha_init.float()).norm().item()
            d_alpha_rel = d_alpha / (alpha_init.float().norm().item() + 1e-8)
        else:
            d_alpha = d_alpha_rel = 0.0

        if w_bar_init is not None:
            d_wbar = (w_bar.float() - w_bar_init.float()).norm().item()
            d_wbar_rel = d_wbar / (w_bar_init.float().norm().item() + 1e-8)
        else:
            d_wbar = d_wbar_rel = 0.0

        snapshots.append(dict(epoch=ep, acc=acc, B=B, coeffs=coeffs, w_bar=w_bar,
                               D=D, pcts=pcts, d_alpha_rel=d_alpha_rel,
                               d_wbar_rel=d_wbar_rel))
        return B, coeffs, w_bar

    # Epoch 0: random init
    B0, coeffs0, w_bar0 = snapshot(0, 0.0)
    alpha_init = coeffs0.clone()
    w_bar_init  = w_bar0.clone()

    for ep in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
        acc = correct / total
        model.eval()
        if ep in checkpoints:
            snapshot(ep, acc)

    # Compute B_final and fill sa_final retroactively
    B_final = snapshots[-1]['B']
    for s in snapshots:
        s['sa_final'] = subspace_alignment(s['B'], B_final)

    for s in snapshots:
        print(f"  {s['epoch']:>4} {s['acc']:>6.3f} "
              f"{s['pcts'].mean():>8.1f} {s['sa_final']:>10.6f} "
              f"{s['d_wbar_rel']:>10.4f} {s['d_alpha_rel']:>10.4f}")

    # --- Analysis ---
    print()
    print("=" * 62)
    print("DYNAMICS ANALYSIS")
    print("=" * 62)

    # When does B converge? (sa_final > 0.9)
    converge_ep = None
    for s in snapshots:
        if s['sa_final'] > 0.9:
            converge_ep = s['epoch']
            break
    if converge_ep is not None:
        print(f"B subspace converges to final (sa>0.9) at epoch {converge_ep}")
    else:
        sa_vals = [s['sa_final'] for s in snapshots]
        print(f"B subspace did not converge (max sa={max(sa_vals):.4f})")

    # When does spectral percentile exceed 70th?
    spec_emerge = None
    for s in snapshots:
        if s['pcts'].mean() > 70:
            spec_emerge = s['epoch']
            break
    if spec_emerge is not None:
        print(f"Spectral elevation (>70th) first seen at epoch {spec_emerge}")
    else:
        print("Spectral elevation >70th not reached")

    # Norm dynamics
    final_d_alpha = snapshots[-1]['d_alpha_rel']
    final_d_wbar  = snapshots[-1]['d_wbar_rel']
    print(f"Final ||Δα||/||α₀|| = {final_d_alpha:.4f}")
    print(f"Final ||ΔW̄||/||W̄₀|| = {final_d_wbar:.4f}")

    if final_d_alpha > 2 * final_d_wbar:
        print("α drifts more than W_bar — consistent with α encoding task-specific variation")
    elif final_d_wbar > 2 * final_d_alpha:
        print("W_bar drifts more than α — W_bar carries task adaptation, α stable")
    else:
        print("α and W_bar drift at similar rates")

    # Phase classification
    if converge_ep is not None and spec_emerge is not None:
        final_acc = snapshots[-1]['acc']
        acc_at_converge = next(s['acc'] for s in snapshots if s['epoch'] == converge_ep)
        if converge_ep <= epochs // 4 and acc_at_converge < 0.5 * final_acc:
            phase = "EARLY LOCK-IN — structure stabilizes before task is learned"
        elif abs(converge_ep - spec_emerge) < 3:
            phase = "CO-EMERGENCE — structure and accuracy rise together"
        else:
            phase = "SEQUENTIAL — one dimension stabilizes before the other"
        print(f"Training phase: {phase}")

    return snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--C",           type=int, default=32)
    parser.add_argument("--n_samples",   type=int, default=2000)
    parser.add_argument("--n_grad",      type=int, default=32)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    run(args.epochs, args.C, args.n_samples, args.n_grad,
        args.checkpoints, args.device, args.seed)
