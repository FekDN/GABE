# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_continual.py — Experiment 21: Continual Learning Chain
#
# PURPOSE:
#   Tests the core GABE claim for continual learning:
#     Freeze (W_bar, B_k). Train only α_i per task.
#     → No catastrophic forgetting by design, since old α_i are never touched.
#
#   Comparison:
#     - GABE-CL:  freeze W_bar, B_k; allocate new α per task
#     - FULL-FT:  full fine-tuning (upper bound on accuracy, lower bound on retention)
#     - FROZEN:   freeze all weights (lower bound on new-task accuracy)
#
#   Task sequence: CIFAR-10 split into 5 binary tasks (pairs of classes).
#     Task 1: {0,1}  Task 2: {2,3}  Task 3: {4,5}  Task 4: {6,7}  Task 5: {8,9}
#
#   Metrics:
#     - Accuracy on each task after training on all subsequent tasks
#     - Average accuracy (across all tasks at end of sequence)
#     - Forgetting: accuracy drop on Task T after training on T+1, T+2, ...
#
# NOTE:
#   This is a proof-of-concept with a small model and few tasks.
#   Standard benchmarks (Split-MNIST, Permuted-MNIST, Split-CIFAR-100)
#   require additional setup beyond the scope of this script.
#
# USAGE:
#   python GABEtest_continual.py
#   python GABEtest_continual.py --n_tasks 5 --epochs_per_task 10 --C 32

import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import copy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Small ConvNet for binary classification
# ---------------------------------------------------------------------------

class SmallConvNet(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.stem = nn.Conv2d(3, C, 3, padding=1)
        self.c1 = nn.Conv2d(C, C, 3, padding=1)
        self.c2 = nn.Conv2d(C, C, 3, padding=1)
        self.c3 = nn.Conv2d(C, C, 3, padding=1)
        self.c4 = nn.Conv2d(C, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Linear(C * 16, 2)  # binary
        self._C = C

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for l in [self.c1, self.c2, self.c3, self.c4]:
            x = torch.relu(l(x))
        return self.head(self.pool(x).flatten(1))

    def gabe_layers(self):
        return [self.c1, self.c2, self.c3, self.c4]


# ---------------------------------------------------------------------------
# GABE-CL model: stores one α per task, shares W_bar and B_k
# ---------------------------------------------------------------------------

class GABECLModel(nn.Module):
    def __init__(self, C=32, n_basis_max=3):
        super().__init__()
        self.backbone = SmallConvNet(C)
        self._C = C
        self._K = n_basis_max
        self._w_bar = {}     # {shape: tensor}
        self._basis = {}     # {shape: (K, D) tensor}
        self._alphas = {}    # {task_id: {shape: (L, K) tensor}}
        self._initialized = False

    def _init_from_model(self, reference_model: SmallConvNet):
        """Decompose reference model's weights into W_bar + B_k."""
        gabe = GABE()
        layers = reference_model.gabe_layers()
        ws = [l.weight.detach().clone() for l in layers]
        shape = tuple(ws[0].shape)

        _, B_s, coeffs, _ = gabe._extract_svd_components(ws)
        K = B_s.shape[0];  D = B_s[0].numel()

        self._w_bar[shape] = torch.stack(ws).mean(0)  # true W_bar: mean across all layers
        self._basis[shape] = B_s             # (K, *shape)
        self._K = K
        self._shape = shape
        self._D = D
        self._initialized = True
        print(f"  GABE-CL init: shape={shape}, K={K}, D={D}")
        return coeffs   # initial task 0 coefficients

    def add_task(self, task_id: int, init_coeffs=None):
        shape = self._shape
        K = self._K
        L = 4   # always 4 group layers
        if init_coeffs is not None:
            alpha = nn.Parameter(init_coeffs.clone())
        else:
            alpha = nn.Parameter(torch.zeros(L, K))
        self._alphas[task_id] = nn.ParameterDict(
            {"alpha": alpha}
        )
        # Register as a proper module so optimizer can find it
        setattr(self, f"task_{task_id}_alpha", self._alphas[task_id])

    def reconstruct_weights(self, task_id: int):
        shape = self._shape
        K = self._K
        w_bar = self._w_bar[shape].to(next(self.parameters()).device)
        basis  = self._basis[shape].to(next(self.parameters()).device)  # (K, *shape)
        alpha  = self._alphas[task_id]["alpha"]                          # (L, K)

        D = w_bar.numel()  # total elements in one weight tensor
        B_flat = basis.view(K, D)   # (K, D)
        w_bar_flat = w_bar.reshape(-1)   # (D,)

        # W_i = W_bar + sum_k alpha[i,k] * B_k
        ws = []
        for i in range(alpha.shape[0]):
            w = w_bar_flat + (alpha[i].unsqueeze(1) * B_flat).sum(0)
            ws.append(w.view(shape))
        return ws

    def forward(self, x, task_id: int):
        ws = self.reconstruct_weights(task_id)
        layers = self.backbone.gabe_layers()
        # Temporarily inject weights
        originals = [l.weight.data.clone() for l in layers]
        for l, w in zip(layers, ws):
            l.weight.data.copy_(w)
        out = self.backbone(x)
        for l, w in zip(layers, originals):
            l.weight.data.copy_(w)
        return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_binary_loader(cifar, class_a, class_b, n_per_class=200, batch_size=32, seed=0):
    idx = [i for i, (_, y) in enumerate(cifar) if y in (class_a, class_b)][:n_per_class*2]
    labels = {class_a: 0, class_b: 1}
    class BinaryWrapper(torch.utils.data.Dataset):
        def __init__(self, base, indices, label_map):
            self.base = base; self.idx = indices; self.lmap = label_map
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            x, y = self.base[self.idx[i]]
            return x, self.lmap[y]
    ds = BinaryWrapper(cifar, idx, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))


def evaluate_binary(model, loader, task_id, device, use_gabe=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if use_gabe:
                out = model(x, task_id)
            else:
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_tasks=5, epochs_per_task=10, C=32, n_per_class=200, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 21: Continual Learning Chain")
    print("=" * 62)
    print(f"n_tasks={n_tasks}  epochs_per_task={epochs_per_task}  C={C}")
    print()

    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf)

    task_pairs = [(2*i, 2*i+1) for i in range(n_tasks)]
    loaders = [make_binary_loader(cifar, a, b, n_per_class, seed=seed)
               for a, b in task_pairs]
    loss_fn = nn.CrossEntropyLoss()

    # --- Condition 1: GABE-CL ---
    print("[GABE-CL] Initializing from pretrained reference model...")
    ref_model = SmallConvNet(C).to(device)
    # Pre-train reference on Task 0
    opt_ref = optim.Adam(ref_model.parameters(), lr=1e-3)
    for _ in range(epochs_per_task):
        for x, y in loaders[0]:
            x, y = x.to(device), y.to(device)
            opt_ref.zero_grad(); loss_fn(ref_model(x), y).backward(); opt_ref.step()

    gabe_model = GABECLModel(C=C).to(device)
    init_coeffs = gabe_model._init_from_model(ref_model)
    gabe_model.add_task(0, init_coeffs)

    gabe_accs = {}   # {task_id: [acc after task 0, after task 1, ...]}
    for t in range(n_tasks):
        if t > 0:
            gabe_model.add_task(t)
        # Train only the current task's alpha
        opt_params = list(gabe_model._alphas[t].parameters())
        opt = optim.Adam(opt_params, lr=1e-2)
        for ep in range(epochs_per_task):
            for x, y in loaders[t]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); loss_fn(gabe_model(x, t), y).backward(); opt.step()

        # Evaluate all tasks seen so far
        for prev_t in range(t + 1):
            acc = evaluate_binary(gabe_model, loaders[prev_t], prev_t,
                                   device, use_gabe=True)
            gabe_accs.setdefault(prev_t, []).append(acc)
        print(f"  After task {t}: {' | '.join(f'T{i}={gabe_accs[i][-1]:.3f}' for i in range(t+1))}")

    # --- Condition 2: Full fine-tune ---
    print("\n[FULL-FT] Full fine-tuning baseline...")
    ft_model = SmallConvNet(C).to(device)
    ft_accs = {}
    for t in range(n_tasks):
        opt = optim.Adam(ft_model.parameters(), lr=1e-3)
        for ep in range(epochs_per_task):
            for x, y in loaders[t]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); loss_fn(ft_model(x), y).backward(); opt.step()
        for prev_t in range(t + 1):
            acc = evaluate_binary(ft_model, loaders[prev_t], prev_t, device)
            ft_accs.setdefault(prev_t, []).append(acc)
        print(f"  After task {t}: {' | '.join(f'T{i}={ft_accs[i][-1]:.3f}' for i in range(t+1))}")

    # --- Summary ---
    print()
    print("=" * 62)
    print("FINAL ACCURACY (after training on all tasks)")
    print("=" * 62)
    print(f"  {'Task':<8} {'GABE-CL':>10} {'FULL-FT':>10} {'ΔForget (FULL)':>16}")
    for t in range(n_tasks):
        ga = gabe_accs[t][-1]
        fa = ft_accs[t][-1]
        peak_fa = max(ft_accs[t])
        forget_ft = peak_fa - fa
        print(f"  Task {t}   {ga:>10.4f} {fa:>10.4f} {-forget_ft:>+16.4f}")

    # Compute average forgetting
    gabe_avg = np.mean([gabe_accs[t][-1] for t in range(n_tasks)])
    ft_avg   = np.mean([ft_accs[t][-1]   for t in range(n_tasks)])
    ft_forget = np.mean([max(ft_accs[t]) - ft_accs[t][-1] for t in range(n_tasks)])
    gabe_forget = np.mean([max(gabe_accs[t]) - gabe_accs[t][-1] for t in range(n_tasks)])

    print()
    print(f"Average accuracy — GABE-CL : {gabe_avg:.4f}")
    print(f"Average accuracy — FULL-FT : {ft_avg:.4f}")
    print(f"Average forgetting — FULL-FT  : {ft_forget:.4f}")
    print(f"Average forgetting — GABE-CL  : {gabe_forget:.4f}")
    print()

    if gabe_forget < ft_forget - 0.05:
        verdict = "GABE-CL significantly reduces forgetting vs full fine-tuning."
    elif gabe_forget < ft_forget:
        verdict = "GABE-CL shows slightly less forgetting."
    else:
        verdict = "No clear forgetting advantage for GABE-CL at this scale."
    print(f"Verdict: {verdict}")
    return dict(gabe_accs=gabe_accs, ft_accs=ft_accs,
                gabe_forget=gabe_forget, ft_forget=ft_forget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks",          type=int, default=5)
    parser.add_argument("--epochs_per_task",  type=int, default=10)
    parser.add_argument("--C",                type=int, default=32)
    parser.add_argument("--n_per_class",      type=int, default=200)
    parser.add_argument("--device",           type=str, default="cpu")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()
    run(args.n_tasks, args.epochs_per_task, args.C,
        args.n_per_class, args.device, args.seed)
