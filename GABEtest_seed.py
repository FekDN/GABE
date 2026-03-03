# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_seed.py — Experiment 13: Seed Reproducibility
#
# PURPOSE:
#   Tests whether the GABE basis subspace span(B) is stable across independent
#   training runs of the same architecture from different random initializations.
#
#   CRITICAL FALSIFIER: If B_k directions vary randomly across seeds
#   (subspace_alignment << 1.0) → the basis is initialization-dependent
#   and the "universal address space" claim does not hold for trained models.
#
# METHOD:
#   1. Train N_SEEDS instances of SmallConvNet on a CIFAR-10 subset.
#   2. Extract GABE basis from each trained model.
#   3. Compute pairwise:
#      (a) Subspace alignment  [0=orthogonal, 1=identical]
#      (b) Max cosine similarity of individual B_k vectors (rotation-aware)
#      (c) Element-wise Pearson r (detects both identity and sign flip)
#   4. Compare against a random orthonormal baseline.
#
# USAGE:
#   python GABEtest_seed.py
#   python GABEtest_seed.py --n_seeds 5 --epochs 30 --device cpu

import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from itertools import combinations
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


class SmallConvNet(nn.Module):
    """4 same-shape conv layers  → GABE group of size 4.  Runs on CIFAR-10."""
    def __init__(self, C: int = 32):
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
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = torch.relu(self.c4(x))
        return self.head(self.pool(x).flatten(1))


def train_model(seed, epochs, n_samples, C, device):
    torch.manual_seed(seed);  np.random.seed(seed)
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, list(range(n_samples))),
        batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    model = SmallConvNet(C).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model.eval()


def extract_basis(model, device):
    gabe = GABE()
    ws = [m.weight.detach().to(device)
          for m in [model.c1, model.c2, model.c3, model.c4]]
    _, B, _, _ = gabe._extract_svd_components(ws)
    K, D = B.shape[0], B[0].numel()
    Q, _ = torch.linalg.qr(B.view(K, D).T.float())
    return Q[:, :K]


def subspace_alignment(B1, B2):
    _, S, _ = torch.linalg.svd(B1.T @ B2)
    return (S ** 2).mean().item()


def max_cosine_per_vector(B1, B2):
    """For each column of B1, max |cos| with any column of B2."""
    return (B1.T @ B2).abs().max(dim=1).values.numpy()


def run(n_seeds=5, epochs=20, n_samples=2000, C=32, device="cpu"):
    print("=" * 62)
    print("GABE Experiment 13: Seed Reproducibility")
    print("=" * 62)
    print(f"n_seeds={n_seeds}  epochs={epochs}  n_samples={n_samples}  C={C}")
    print()

    bases = []
    for seed in range(n_seeds):
        print(f"  [seed {seed}] training...", end=" ", flush=True)
        m = train_model(seed, epochs, n_samples, C, device)
        B = extract_basis(m, device)
        bases.append(B)
        print(f"B.shape={B.shape}")

    D, K = bases[0].shape
    pairs = list(combinations(range(n_seeds), 2))
    sa_vals, mc_vals = [], []

    print(f"\nPairwise metrics ({len(pairs)} pairs):  D={D}  K={K}")
    print(f"  {'Pair':<8} {'SubspaceAlign':>14} {'MaxCos(mean)':>14} {'MaxCos(min)':>12}")
    print("  " + "-" * 52)
    for i, j in pairs:
        sa = subspace_alignment(bases[i], bases[j])
        mc = max_cosine_per_vector(bases[i], bases[j])
        sa_vals.append(sa)
        mc_vals.append(mc.mean())
        print(f"  ({i},{j})     {sa:>14.6f} {mc.mean():>14.4f} {mc.min():>12.4f}")

    # Random baseline: expected subspace alignment = K/D
    rand_sa = []
    for _ in range(300):
        Q1, _ = torch.linalg.qr(torch.randn(D, K));  Q1 = Q1[:, :K]
        Q2, _ = torch.linalg.qr(torch.randn(D, K));  Q2 = Q2[:, :K]
        rand_sa.append(subspace_alignment(Q1, Q2))
    rand_mean = np.mean(rand_sa)

    sa_mean = np.mean(sa_vals)
    print()
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"SubspaceAlignment  : {sa_mean:.6f} ± {np.std(sa_vals):.6f}")
    print(f"Random baseline    : {rand_mean:.6f}  (theoretical K/D = {K/D:.6f})")
    print(f"Elevation vs random: {sa_mean / (rand_mean + 1e-12):.2f}×")
    print(f"MaxCosine (mean)   : {np.mean(mc_vals):.4f}")
    print()

    if sa_mean > 0.7:
        verdict = "STABLE — span(B) highly consistent across seeds."
    elif sa_mean > 3 * rand_mean:
        verdict = "PARTIALLY STABLE — subspace elevated vs random, not fully converged."
    else:
        verdict = "UNSTABLE — span(B) ~ random. Initialization-dependent."

    if np.mean(mc_vals) > 0.8 and sa_mean > 0.7:
        rot_note = "Individual B_k vectors also co-align (strong structural lock-in)."
    elif sa_mean > 0.5 and np.mean(mc_vals) < 0.5:
        rot_note = "Subspace stable, individual vectors rotate — consistent with CKA=1.0 semantics."
    else:
        rot_note = "No strong individual vector alignment."

    print(f"Verdict     : {verdict}")
    print(f"Rotation    : {rot_note}")
    return dict(sa_vals=sa_vals, mc_vals=mc_vals, rand_mean=rand_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds",   type=int, default=5)
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--C",         type=int, default=32)
    parser.add_argument("--device",    type=str, default="cpu")
    args = parser.parse_args()
    run(args.n_seeds, args.epochs, args.n_samples, args.C, args.device)
