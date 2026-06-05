# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_epoch_match.py  —  Experiment 28: Epoch-Matched Cross-Architecture Alignment
#
# QUESTION:
#   How much of the cross-architecture alignment gap measured in Exp 27 (Steps 1-6)
#   is an artefact of comparing models at identical epoch counts, when in reality
#   different architectures crystallise their GABE basis at different rates?
#
# MOTIVATION:
#   Exp 27 Step 7 found that Plain vs Skip with RMSprop has +35.7% higher alignment
#   when compared at (eA=1, eB=10) instead of (eA=15, eB=15). This implies the
#   measured arch gap is inflated when one architecture is "more mature" than the other.
#   Exp 28 systematises this finding:
#     - Fine-grained epoch grid (up to max_epochs, dense early)
#     - Bootstrap significance test for off-diagonal alignment gain
#     - Correction factor: how much is the equal-epoch gap inflated?
#     - Protocol recommendation table for future cross-arch experiments
#
# DESIGN:
#   For each (arch_A, arch_B, opt) triplet:
#     1. Train both from SAME seed at same opt, checkpoint at every epoch
#     2. Compute full M[eA, eB] alignment matrix
#     3. Find (eA*, eB*) = argmax M
#     4. Bootstrap test: is max(off-diag) > max(diag) at p < 0.05?
#     5. Compute correction factor = M[eA*, eB*] / max(diag)
#     6. Report the "fair comparison epoch" for each pair
#
# METRICS:
#   - span(B) subspace alignment (main metric)
#   - wbar_cos (secondary)
#   - Correction factor (off-diag peak / diag peak)
#   - Bootstrap p-value for off-diagonal advantage
#   - Convergence epoch per arch (align to own final > 0.95)
#
# USAGE:
#   python GABEtest_epoch_match.py
#   python GABEtest_epoch_match.py --max_epochs 30 --C 32 --n_samples 2000
#   python GABEtest_epoch_match.py --max_epochs 30 --C 64 --device cuda

import sys, os, argparse, itertools, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ─────────────────────────────────────────────────────────────────────────────
# Architectures (same as Exp 26b / Exp 27 for direct comparability)
# ─────────────────────────────────────────────────────────────────────────────

class NetPlain(nn.Module):
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
    def gabe_group(self):
        return [self.c1, self.c2, self.c3, self.c4]

class NetBN(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.stem = nn.Conv2d(3, C, 3, padding=1)
        self.c1 = nn.Conv2d(C, C, 3, padding=1); self.bn1 = nn.BatchNorm2d(C)
        self.c2 = nn.Conv2d(C, C, 3, padding=1); self.bn2 = nn.BatchNorm2d(C)
        self.c3 = nn.Conv2d(C, C, 3, padding=1); self.bn3 = nn.BatchNorm2d(C)
        self.c4 = nn.Conv2d(C, C, 3, padding=1); self.bn4 = nn.BatchNorm2d(C)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Linear(C * 16, 10)
    def forward(self, x):
        x = torch.relu(self.stem(x))
        for l, bn in zip([self.c1, self.c2, self.c3, self.c4],
                         [self.bn1, self.bn2, self.bn3, self.bn4]):
            x = torch.relu(bn(l(x)))
        return self.head(self.pool(x).flatten(1))
    def gabe_group(self):
        return [self.c1, self.c2, self.c3, self.c4]

class NetSkip(nn.Module):
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
            x = torch.relu(l(x) + x)
        return self.head(self.pool(x).flatten(1))
    def gabe_group(self):
        return [self.c1, self.c2, self.c3, self.c4]

ARCH_CLASSES = {"Plain": NetPlain, "BN": NetBN, "Skip": NetSkip}

OPTIMIZER_DEFAULTS = {
    "AdamW":   dict(lr=1e-3, weight_decay=1e-4),
    "SGD":     dict(lr=1e-2, momentum=0.9, weight_decay=1e-4),
    "RMSprop": dict(lr=1e-3, momentum=0.9, weight_decay=1e-4, alpha=0.99),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_optimizer(name, params):
    cfg = OPTIMIZER_DEFAULTS[name]
    if name == "AdamW":
        return optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif name == "SGD":
        return optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"],
                         weight_decay=cfg["weight_decay"])
    elif name == "RMSprop":
        return optim.RMSprop(params, lr=cfg["lr"], momentum=cfg["momentum"],
                             weight_decay=cfg["weight_decay"], alpha=cfg["alpha"])
    raise ValueError(name)

def extract_gabe(model):
    """Returns (w_bar_flat, B_flat_normalised, K, D)."""
    ws = [l.weight.detach().cpu() for l in model.gabe_group()]
    g = GABE()
    w_bar, B, _, shape = g._extract_svd_components(ws)
    D = int(np.prod(shape[1:]))
    w_flat = w_bar.reshape(-1).to(torch.float64)
    B_flat = B.view(B.shape[0], -1).to(torch.float64)
    B_flat = B_flat / B_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return w_flat, B_flat, B.shape[0], D

def span_align(B1, B2):
    K = min(B1.shape[0], B2.shape[0])
    gram = B1[:K] @ B2[:K].T
    return (gram ** 2).sum().item() / K

def wbar_cos(w1, w2):
    return (w1 @ w2 / (w1.norm() * w2.norm()).clamp(min=1e-12)).item()

def rand_baseline(K, D):
    return K / D

def hline(n=76): print("─" * n)


# ─────────────────────────────────────────────────────────────────────────────
# Training with per-epoch checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def train_all_epochs(arch_cls, C, loader, max_epochs, device, opt_name, seed):
    """Train and save GABE snapshot at EVERY epoch (dense grid)."""
    set_seed(seed)
    model = arch_cls(C).to(device)
    opt   = make_optimizer(opt_name, model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    snapshots = {}
    acc_log   = {}
    for ep in range(1, max_epochs + 1):
        model.train()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss_fn(out, y).backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
        w, B, K, D = extract_gabe(model)
        snapshots[ep] = (w, B, K, D)
        acc_log[ep]   = correct / max(total, 1)
    return snapshots, acc_log


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap significance test
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_offdiag_test(M_ratio, n_bootstrap=1000, alpha=0.05):
    """
    H0: max(off-diag) <= max(diag)
    Test by permuting the row/column labels and checking the null distribution
    of (max_off_diag - max_diag) under random epoch assignments.

    Returns (off_diag_gain, p_value, significant).
    """
    n = M_ratio.shape[0]
    diag_vals    = [M_ratio[i, i] for i in range(n)]
    off_diag     = M_ratio.copy()
    np.fill_diagonal(off_diag, np.nan)
    max_diag  = float(np.nanmax(diag_vals))
    max_off   = float(np.nanmax(off_diag))
    obs_stat  = max_off - max_diag   # observed test statistic

    # Null distribution: permute epoch labels for one side, re-compute
    null_stats = []
    for _ in range(n_bootstrap):
        perm = np.random.permutation(n)
        M_perm = M_ratio[perm, :]     # permute rows
        d_perm = [M_perm[i, i] for i in range(n)]
        o_perm = M_perm.copy()
        np.fill_diagonal(o_perm, np.nan)
        null_stats.append(float(np.nanmax(o_perm)) - float(np.nanmax(d_perm)))

    null_stats = np.array(null_stats)
    p_value = float(np.mean(null_stats >= obs_stat))
    return obs_stat, p_value, p_value < alpha


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run(max_epochs=20, C=32, n_samples=2000, device="cpu",
        base_seed=42, n_seeds=3,
        opt_names=None, n_bootstrap=500):

    if opt_names is None:
        opt_names = ["AdamW", "SGD", "RMSprop"]

    print("=" * 76)
    print("GABE Experiment 28: Epoch-Matched Cross-Architecture Alignment")
    print("=" * 76)
    print(f"  max_epochs={max_epochs}  C={C}  n_samples={n_samples}")
    print(f"  n_seeds={n_seeds}  device={device}  optimizers={opt_names}")
    print(f"  n_bootstrap={n_bootstrap}")
    print()
    print("  QUESTION: How much of the cross-arch gap is epoch-mismatch artefact?")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    try:
        ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                          download=True, transform=tf)
        sub = torch.utils.data.Subset(ds, list(range(min(n_samples, len(ds)))))
        loader = torch.utils.data.DataLoader(sub, batch_size=64,
                                             shuffle=False, num_workers=0)
        print("  Data: CIFAR-10")
    except Exception as e:
        print(f"  CIFAR-10 unavailable ({e}). Using synthetic data.")
        set_seed(base_seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(n_samples, 3, 32, 32),
                torch.randint(0, 10, (n_samples,))),
            batch_size=64, shuffle=False, num_workers=0)

    arch_pairs = list(itertools.combinations(list(ARCH_CLASSES.keys()), 2))
    epoch_list = list(range(1, max_epochs + 1))

    # ── Training ──────────────────────────────────────────────────────────────
    hline()
    print("PHASE 1 — Training all (arch × opt × seed) with per-epoch checkpoints")
    hline()
    print()

    # {(arch, opt, seed): snapshots_dict}
    all_snaps = {}
    all_acc   = {}

    for arch_name, arch_cls in ARCH_CLASSES.items():
        for opt_name in opt_names:
            for si in range(n_seeds):
                seed_i = base_seed + si * 37
                key    = (arch_name, opt_name, si)
                t0 = time.time()
                print(f"  [{arch_name:<7} {opt_name:<8} s={si}] ...", end=" ", flush=True)
                snaps, acc = train_all_epochs(arch_cls, C, loader,
                                             max_epochs, device, opt_name, seed_i)
                all_snaps[key] = snaps
                all_acc[key]   = acc
                print(f"done ({time.time()-t0:.0f}s)  "
                      f"acc@ep{max_epochs}={acc[max_epochs]:.3f}")

    K = all_snaps[(list(ARCH_CLASSES.keys())[0], opt_names[0], 0)][1][2]
    D = all_snaps[(list(ARCH_CLASSES.keys())[0], opt_names[0], 0)][1][3]
    rb = rand_baseline(K, D)
    print(f"\n  K={K}  D={D}  rand_baseline={rb:.4e}")

    # ── Per-arch convergence speed ─────────────────────────────────────────────
    hline()
    print("PHASE 2 — Per-arch convergence speed (align to own final basis, thr=0.95)")
    hline()
    print()
    print(f"  {'Arch':<8} {'Opt':<9}  {'conv_ep':>8}  "
          f"{'sa@ep1':>8}  {'sa@mid':>8}  {'sa@final':>9}  {'acc@final':>10}")
    print("  " + "-" * 64)

    conv_epochs = {}   # {(arch, opt): mean conv ep}
    for arch_name in ARCH_CLASSES:
        for opt_name in opt_names:
            conv_list, sa1_list, sam_list, acc_list = [], [], [], []
            for si in range(n_seeds):
                snaps  = all_snaps[(arch_name, opt_name, si)]
                final_B = snaps[max_epochs][1]
                conv_ep = max_epochs
                for ep in epoch_list:
                    sa = span_align(snaps[ep][1], final_B)
                    if sa > 0.95 and conv_ep == max_epochs:
                        conv_ep = ep
                conv_list.append(conv_ep)
                sa1_list.append(span_align(snaps[1][1], final_B))
                mid = max_epochs // 2
                sam_list.append(span_align(snaps[mid][1], final_B))
                acc_list.append(all_acc[(arch_name, opt_name, si)][max_epochs])

            mu_conv = float(np.mean(conv_list))
            conv_epochs[(arch_name, opt_name)] = mu_conv
            print(f"  {arch_name:<8} {opt_name:<9}  {mu_conv:>8.1f}  "
                  f"{np.mean(sa1_list):>8.3f}  {np.mean(sam_list):>8.3f}  "
                  f"{'1.000':>9}  {np.mean(acc_list):>10.3f}")
    print()

    # ── Full alignment matrices ────────────────────────────────────────────────
    hline()
    print("PHASE 3 — Full M[eA, eB] alignment matrices + bootstrap test")
    hline()

    # Summary table rows accumulated
    summary_rows = []

    for arch_A, arch_B in arch_pairs:
        print()
        print(f"  ══ {arch_A} vs {arch_B} ══")
        print()

        for opt_name in opt_names:
            # Average M over seeds
            M_list = []
            for si in range(n_seeds):
                snaps_A = all_snaps[(arch_A, opt_name, si)]
                snaps_B = all_snaps[(arch_B, opt_name, si)]
                M = np.zeros((max_epochs, max_epochs))
                for i, eA in enumerate(epoch_list):
                    for j, eB in enumerate(epoch_list):
                        M[i, j] = span_align(snaps_A[eA][1], snaps_B[eB][1]) / rb
                M_list.append(M)
            M_mean = np.mean(M_list, axis=0)

            # Diagonal analysis
            diag      = [M_mean[i, i] for i in range(max_epochs)]
            diag_peak = float(np.max(diag))
            diag_ep   = epoch_list[int(np.argmax(diag))]
            diag_min  = float(np.min(diag))
            diag_min_ep = epoch_list[int(np.argmin(diag))]

            # Off-diagonal peak
            M_off  = M_mean.copy()
            np.fill_diagonal(M_off, np.nan)
            peak_flat = int(np.nanargmax(M_off))
            peak_i, peak_j = divmod(peak_flat, max_epochs)
            peak_val  = float(M_off[peak_i, peak_j])
            peak_eA   = epoch_list[peak_i]
            peak_eB   = epoch_list[peak_j]

            gain_pct  = (peak_val - diag_peak) / max(diag_peak, 1e-9) * 100

            # Bootstrap test
            obs_stat, p_val, sig = bootstrap_offdiag_test(M_mean, n_bootstrap)

            # Convergence ratio
            cA = conv_epochs.get((arch_A, opt_name), float("nan"))
            cB = conv_epochs.get((arch_B, opt_name), float("nan"))
            faster = (arch_B if cA > cB + 0.5 else
                      arch_A if cB > cA + 0.5 else "≈equal")

            # Print compact matrix (show every 5th epoch + first + last)
            show_eps = sorted(set([1, 2, 3, 5] +
                                  list(range(5, max_epochs + 1, 5)) +
                                  [max_epochs]))
            show_idx = [epoch_list.index(e) for e in show_eps if e in epoch_list]

            header = "    eA\\eB" + "".join(
                f"  {str(show_eps[j]):>4}" for j in range(len(show_idx)))
            print(f"  [{opt_name}]")
            print(header)
            for i in show_idx:
                eA  = epoch_list[i]
                row = f"    {str(eA):>5}"
                for j in show_idx:
                    v    = M_mean[i, j]
                    mark = "*" if i == j else " "
                    row += f"  {v:>3.0f}{mark}"
                print(row)
            print(f"    (* = diagonal  |  values = alignment / rand_baseline)")
            print()
            print(f"    Diag peak:    {diag_peak:>6.0f}× at ep{diag_ep}")
            print(f"    Diag min:     {diag_min:>6.0f}× at ep{diag_min_ep}  "
                  f"(drift = {(diag_peak-diag_min)/max(diag_peak,1e-9)*100:.1f}%)")
            print(f"    Off-diag peak:{peak_val:>6.0f}× at (eA={peak_eA}, eB={peak_eB})")
            gain_str = f"{gain_pct:+.1f}%"
            sig_str  = f"p={p_val:.3f} {'*' if sig else ''}"
            print(f"    Off-diag gain:{gain_pct:>+7.1f}%  [{sig_str}]")
            if gain_pct > 5 and sig:
                print(f"    → {faster} converges faster. "
                      f"Fair epoch: stop {arch_A} at ep{peak_eA}, {arch_B} at ep{peak_eB}.")
            elif gain_pct > 5 and not sig:
                print(f"    → Off-diag advantage not significant (p={p_val:.3f}).")
            else:
                print(f"    → Equal-epoch comparison is near-optimal.")
            print()

            summary_rows.append({
                "arch_A": arch_A, "arch_B": arch_B, "opt": opt_name,
                "diag_peak": diag_peak, "diag_ep": diag_ep,
                "off_peak": peak_val, "peak_eA": peak_eA, "peak_eB": peak_eB,
                "gain_pct": gain_pct, "p_val": p_val, "sig": sig,
                "faster": faster,
                "conv_A": conv_epochs.get((arch_A, opt_name), float("nan")),
                "conv_B": conv_epochs.get((arch_B, opt_name), float("nan")),
            })

    # ── Summary protocol table ────────────────────────────────────────────────
    hline()
    print("PHASE 4 — Protocol recommendation table")
    hline()
    print()
    print("  For each (arch pair, opt): recommended fair comparison epoch and")
    print("  magnitude of equal-epoch alignment underestimation.")
    print()
    print(f"  {'Pair':<20} {'Opt':<9}  {'diag_pk':>8}  {'off_pk':>7}  "
          f"{'gain':>7}  {'p':>6}  {'fair_ep (A,B)':>16}  verdict")
    print("  " + "-" * 88)

    for r in summary_rows:
        pair  = f"{r['arch_A']} vs {r['arch_B']}"
        sig_m = "*" if r["sig"] else " "
        fair  = f"({r['peak_eA']},{r['peak_eB']})" if r["gain_pct"] > 5 else "equal"
        verdict = ("EPOCH-MATCH NEEDED" if r["gain_pct"] > 5 and r["sig"]
                   else "equal-epoch OK")
        print(f"  {pair:<20} {r['opt']:<9}  {r['diag_peak']:>8.0f}  "
              f"{r['off_peak']:>7.0f}  {r['gain_pct']:>+6.1f}%  "
              f"{r['p_val']:>5.3f}{sig_m}  {fair:>16}  {verdict}")

    print()

    # ── Overall verdict ────────────────────────────────────────────────────────
    hline()
    print("VERDICT")
    hline()
    print()
    needs_match = [r for r in summary_rows if r["gain_pct"] > 5 and r["sig"]]
    ok_equal    = [r for r in summary_rows if not (r["gain_pct"] > 5 and r["sig"])]

    if needs_match:
        max_gain = max(r["gain_pct"] for r in needs_match)
        print(f"  {len(needs_match)}/{len(summary_rows)} pairs benefit significantly from epoch-matching.")
        print(f"  Maximum alignment gain from epoch-matching: +{max_gain:.1f}%")
        print()
        print("  RECOMMENDATION: When comparing architectures with RMSprop or other")
        print("  slowly-converging optimizers, stop training when each architecture's")
        print("  GABE basis has converged (span_align to own final > 0.95), not at")
        print("  a fixed wall-clock epoch count.")
        print()
        print("  CORRECTION FACTORS for Exp 27 Step 1 results:")
        for r in needs_match:
            corr = (r["off_peak"] - r["diag_peak"]) / max(r["diag_peak"], 1e-9) * 100
            print(f"    {r['arch_A']} vs {r['arch_B']} [{r['opt']}]: "
                  f"+{corr:.1f}% underestimation → true alignment "
                  f"≈ {r['off_peak']:.0f}× (was {r['diag_peak']:.0f}×)")
    else:
        print(f"  All {len(summary_rows)} pairs: equal-epoch comparison is near-optimal.")
        print("  The arch gaps measured in Exp 27 Steps 1-6 are NOT inflated by")
        print("  epoch mismatch for the tested optimizers.")

    print()
    print("  PROTOCOL RECOMMENDATION FOR FUTURE CROSS-ARCH EXPERIMENTS:")
    print("  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │ 1. Always measure per-arch convergence epoch (align to own final │")
    print("  │    basis > 0.95) before comparing across architectures.          │")
    print("  │ 2. Use the 'fair epoch' = max(conv_A, conv_B) as the stopping    │")
    print("  │    criterion for equal-maturity comparison.                      │")
    print("  │ 3. For AdamW/SGD: equal-epoch comparison is acceptable.          │")
    print("  │ 4. For RMSprop/slow optimizers: epoch-matching is required.      │")
    print("  │ 5. Report both equal-epoch and epoch-matched results.            │")
    print("  └──────────────────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 28: Epoch-Matched Cross-Architecture Alignment")
    parser.add_argument("--max_epochs",  type=int,   default=20)
    parser.add_argument("--C",           type=int,   default=32)
    parser.add_argument("--n_samples",   type=int,   default=2000)
    parser.add_argument("--n_seeds",     type=int,   default=3)
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--base_seed",   type=int,   default=42)
    parser.add_argument("--optimizers",  type=str,   default="AdamW,SGD,RMSprop")
    parser.add_argument("--n_bootstrap", type=int,   default=500)
    args = parser.parse_args()
    run(max_epochs=args.max_epochs, C=args.C, n_samples=args.n_samples,
        device=args.device, base_seed=args.base_seed, n_seeds=args.n_seeds,
        opt_names=[o.strip() for o in args.optimizers.split(",")],
        n_bootstrap=args.n_bootstrap)
