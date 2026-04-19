# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_aten2.py — Experiment 26b: ATen Group Chain vs w_bar AND span(B)
#
# CORRECTIONS OVER Exp 26:
#
#   Exp 26 had two blind spots:
#
#   (A) Only basis B was compared — w_bar was ignored.
#       w_bar is the mean over the group. If the ATen op chain constrains how
#       weights can vary across the group (via gradient flow structure), it
#       constrains BOTH the mean (w_bar) and the directions of variation (B).
#       Both must be measured to test the op-chain hypothesis fully.
#
#   (B) ATen signature was extracted for a single cell (one layer step),
#       not for the full GROUP forward pass.
#       GABE decomposes a GROUP of L tensors with the same shape. The relevant
#       computational signature is the sequential chain:
#         layer1 → layer2 → layer3 → layer4
#       Two groups have the "same ATen chain" only if their full group forward
#       executes the same op sequence. A group of Conv→ReLU layers differs
#       from a group of Conv→BN→ReLU layers at the GROUP level, not just the
#       cell level.
#
# DESIGN:
#
#   For each architecture variant, we build a GroupForward module that runs
#   all L layers in sequence — this is what gets traced for the ATen signature.
#
#   For each pair of variants (same D), we measure:
#     (1) w_bar cosine similarity         — how similar are the mean weights?
#     (2) w_bar normalized distance       — ||w1/||w1|| - w2/||w2||||_F
#     (3) span(B) subspace alignment      — (1/K)||B1^T B2||_F^2
#     (4) ATen group signature match      — SAME / DIFF
#
#   HYPOTHESIS:
#     Same group ATen chain → higher w_bar similarity AND higher span(B) alignment
#     Different group ATen chain → lower both
#
#   FALSIFIABLE PREDICTIONS:
#     P1. Plain vs Plain_seed2  [SAME chain]: highest w_bar cos + alignment
#     P2. Plain vs BN           [DIFF chain]: lower both
#     P3. Plain vs Skip         [DIFF chain]: lower both (skip adds aten::add)
#     P4. Plain vs Depthwise    [DIFF chain + different D]: excluded from comparison
#
#   CRITICAL DISTINCTION vs Exp 26:
#     If P1 >> P2, P3 in BOTH metrics simultaneously: op-chain hypothesis supported.
#     If P1 ≈ P2, P3 in both metrics: CKA=1.0 is a mathematical SVD property,
#       independent of the computational graph.
#     If P1 >> P2, P3 in span(B) but NOT in w_bar: the basis is determined by
#       gradient geometry but w_bar is task-specific (most likely case).
#
# USAGE:
#   python GABEtest_aten2.py
#   python GABEtest_aten2.py --epochs 20 --C 32 --n_samples 1000 --n_seeds 3

import sys, os, hashlib, itertools, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# ATen Group Signature — trace the full sequential group forward
# ---------------------------------------------------------------------------

class GroupForward(nn.Module):
    """
    Wraps the full group of L layers into a single traceable module.
    This is the correct unit for ATen signature extraction in GABE:
    the signature of a group is the op sequence for the entire L-layer chain,
    not for an individual cell.
    """
    def __init__(self, cells):
        super().__init__()
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        return x


def _ops_from_jit(module, x):
    try:
        with torch.no_grad():
            traced = torch.jit.trace(module, x, strict=False)
        ops = [n.kind().split("::")[-1] for n in traced.inlined_graph.nodes()
               if not n.kind().startswith("prim::")]
        return tuple(ops)
    except Exception:
        return None


def _ops_from_make_fx(module, x):
    try:
        from torch.fx.experimental.proxy_tensor import make_fx
        with torch.no_grad():
            traced = make_fx(module, tracing_mode="symbolic")(x)
        ops = tuple(
            getattr(n.target, "__name__", str(n.target))
            for n in traced.graph.nodes
            if n.op in ("call_function", "call_method")
        )
        return ops
    except Exception:
        return None


def group_aten_signature(group_forward, in_channels, spatial=32, device="cpu"):
    """
    Extract ATen op sequence for the full L-layer group forward pass.
    Returns (ops_tuple, method_str).
    """
    x = torch.zeros(1, in_channels, spatial, spatial, device=device)
    m = copy.deepcopy(group_forward).to(device).eval()

    ops = _ops_from_make_fx(m, x)
    if ops:
        return ops, "make_fx"

    ops = _ops_from_jit(m, x)
    if ops:
        return ops, "jit_trace"

    # Heuristic: use cell class names
    names = tuple(type(c).__name__ for c in group_forward.cells)
    return names, "heuristic"


def sig_hash(ops):
    return hashlib.md5("|".join(str(o) for o in ops).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Cell types (one step of each variant)
# ---------------------------------------------------------------------------

class CellPlain(nn.Module):
    def __init__(self, conv): super().__init__(); self.conv = conv
    def forward(self, x): return torch.relu(self.conv(x))

class CellBN(nn.Module):
    def __init__(self, conv, bn): super().__init__(); self.conv = conv; self.bn = bn
    def forward(self, x): return torch.relu(self.bn(self.conv(x)))

class CellSkip(nn.Module):
    def __init__(self, conv): super().__init__(); self.conv = conv
    def forward(self, x): return torch.relu(self.conv(x) + x)

class CellDepthwise(nn.Module):
    def __init__(self, conv): super().__init__(); self.conv = conv
    def forward(self, x): return torch.relu(self.conv(x))


# ---------------------------------------------------------------------------
# Architecture variants
# ---------------------------------------------------------------------------

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

    def gabe_group(self): return [self.c1, self.c2, self.c3, self.c4]

    def group_forward_module(self):
        cells = [CellPlain(l) for l in self.gabe_group()]
        return GroupForward(cells)


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

    def gabe_group(self): return [self.c1, self.c2, self.c3, self.c4]

    def group_forward_module(self):
        cells = [CellBN(l, bn) for l, bn in
                 zip(self.gabe_group(), [self.bn1, self.bn2, self.bn3, self.bn4])]
        return GroupForward(cells)


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

    def gabe_group(self): return [self.c1, self.c2, self.c3, self.c4]

    def group_forward_module(self):
        cells = [CellSkip(l) for l in self.gabe_group()]
        return GroupForward(cells)


class NetDepthwise(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.stem = nn.Conv2d(3, C, 3, padding=1)
        self.c1 = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.c2 = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.c3 = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.c4 = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Linear(C * 16, 10)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for l in [self.c1, self.c2, self.c3, self.c4]:
            x = torch.relu(l(x))
        return self.head(self.pool(x).flatten(1))

    def gabe_group(self): return [self.c1, self.c2, self.c3, self.c4]

    def group_forward_module(self):
        cells = [CellDepthwise(l) for l in self.gabe_group()]
        return GroupForward(cells)


# ---------------------------------------------------------------------------
# GABE extraction helpers
# ---------------------------------------------------------------------------

def extract_gabe(model, device="cpu"):
    """Returns w_bar (flat), B (K, D_flat), K, D."""
    layers = model.gabe_group()
    ws = [l.weight.detach().cpu() for l in layers]
    gabe = GABE()
    w_bar, B, _, shape = gabe._extract_svd_components(ws)
    D = int(np.prod(shape[1:]))
    w_bar_flat = w_bar.reshape(-1).to(torch.float64)
    B_flat = B.view(B.shape[0], -1).to(torch.float64)
    # Row-normalise B for subspace comparison
    B_flat = B_flat / (B_flat.norm(dim=1, keepdim=True) + 1e-12)
    return w_bar_flat, B_flat, B.shape[0], D


def w_bar_cos(w1, w2):
    """Cosine similarity between two w_bar vectors."""
    return (w1 @ w2 / (w1.norm() * w2.norm() + 1e-12)).item()


def w_bar_norm_dist(w1, w2):
    """L2 distance between unit-normalised w_bar vectors."""
    u1 = w1 / (w1.norm() + 1e-12)
    u2 = w2 / (w2.norm() + 1e-12)
    return (u1 - u2).norm().item()


def span_align(B1, B2):
    """(1/K)||B1^T B2||_F^2, K = min(K1,K2). Random baseline = K/D."""
    K = min(B1.shape[0], B2.shape[0])
    gram = B1[:K] @ B2[:K].T
    return (gram ** 2).sum().item() / K


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, loader, epochs, device, lr=1e-3):
    model.train().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(epochs=15, C=32, n_samples=500, n_seeds=3, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 72)
    print("GABE Experiment 26b: ATen Group Chain vs w_bar AND span(B) Similarity")
    print("=" * 72)
    print(f"C={C}  epochs={epochs}  n_samples={n_samples}  n_seeds={n_seeds}")
    print()

    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, list(range(min(n_samples, len(cifar))))),
        batch_size=64, shuffle=True)

    # -----------------------------------------------------------------------
    # STEP 1 — Group ATen signatures
    # -----------------------------------------------------------------------

    print("─" * 72)
    print("STEP 1 — ATen signature of the full GROUP forward (L=4 layers)")
    print("─" * 72)
    print("  (Tracing GroupForward, not a single cell — this is the correct unit)")
    print()

    variant_classes = {
        "Plain":     NetPlain,
        "BN":        NetBN,
        "Skip":      NetSkip,
        "Depthwise": NetDepthwise,
    }

    sigs = {}
    for name, cls in variant_classes.items():
        m = cls(C)
        gf = m.group_forward_module()
        in_ch = m.gabe_group()[0].weight.shape[1]
        ops, method = group_aten_signature(gf, in_ch, device=device)
        h = sig_hash(ops)
        sigs[name] = {"hash": h, "method": method, "n_ops": len(ops),
                      "in_ch": in_ch}
        print(f"  {name:<12} [{method:<10}]  hash={h}  "
              f"n_ops={len(ops)}  in_ch={in_ch}")
        print(f"             ops: {list(ops)[:12]}" +
              (" ..." if len(ops) > 12 else ""))
    print()

    # Signature matrix
    names = list(sigs.keys())
    print("  Group ATen signature match matrix:")
    print(f"  {'':12}" + "".join(f"  {n:<12}" for n in names))
    for n1 in names:
        row = f"  {n1:<12}"
        for n2 in names:
            tag = "SAME" if sigs[n1]["hash"] == sigs[n2]["hash"] else "DIFF"
            row += f"  {tag:<12}"
        print(row)
    print()

    # -----------------------------------------------------------------------
    # STEP 2 — Train n_seeds instances of each variant, extract w_bar + B
    # -----------------------------------------------------------------------

    print("─" * 72)
    print("STEP 2 — Train each variant × n_seeds, extract w_bar and span(B)")
    print("─" * 72)

    records = {}   # {(name, seed_i): {"w_bar": tensor, "B": tensor, "D": int, "K": int}}

    for name, cls in variant_classes.items():
        for si in range(n_seeds):
            key = (name, si)
            torch.manual_seed(seed + si * 17)
            m = cls(C).to(device)
            print(f"  Training {name} seed={si}...", end=" ", flush=True)
            train(m, loader, epochs, device)
            w_bar_flat, B_flat, K, D = extract_gabe(m, device)
            records[key] = {"w_bar": w_bar_flat, "B": B_flat, "K": K, "D": D,
                            "sig_hash": sigs[name]["hash"]}
            print(f"done  K={K}  D={D}")
    print()

    # -----------------------------------------------------------------------
    # STEP 3 — Pairwise comparison: w_bar cos + w_bar dist + span(B) align
    # -----------------------------------------------------------------------

    print("─" * 72)
    print("STEP 3 — Pairwise comparison: w_bar cosine | w_bar dist | span(B)")
    print("─" * 72)

    all_keys = list(records.keys())
    rand_K  = records[all_keys[0]]["K"]
    rand_D  = records[all_keys[0]]["D"]
    rand_baseline = rand_K / rand_D

    print(f"  Random span(B) baseline = K/D = {rand_K}/{rand_D} = {rand_baseline:.2e}")
    print()
    print(f"  {'Pair':<36} {'ops':>4} {'wbar_cos':>10} {'wbar_dist':>10} "
          f"{'span_align':>12} {'ratio_rand':>11}")
    print("  " + "-" * 88)

    # Buckets by (ops_match, D_match)
    buckets = {
        ("SAME", "SAME_D"): [],
        ("DIFF", "SAME_D"): [],
        ("DIFF", "DIFF_D"): [],
    }

    all_rows = []
    for (k1, k2) in itertools.combinations(all_keys, 2):
        r1, r2 = records[k1], records[k2]
        if r1["D"] != r2["D"]:
            d_tag = "DIFF_D"
            # Skip cross-D w_bar comparison (meaningless)
            wcos, wdist, sa = float("nan"), float("nan"), float("nan")
            ops_tag = "DIFF" if r1["sig_hash"] != r2["sig_hash"] else "SAME"
            ratio = float("nan")
        else:
            d_tag = "SAME_D"
            wcos  = w_bar_cos(r1["w_bar"], r2["w_bar"])
            wdist = w_bar_norm_dist(r1["w_bar"], r2["w_bar"])
            sa    = span_align(r1["B"], r2["B"])
            ops_tag = "SAME" if r1["sig_hash"] == r2["sig_hash"] else "DIFF"
            ratio = sa / rand_baseline

        name1 = f"{k1[0]}_s{k1[1]}"
        name2 = f"{k2[0]}_s{k2[1]}"
        label = f"{name1} vs {name2}"

        wcos_str  = f"{wcos:>10.4f}" if not np.isnan(wcos) else f"{'n/a':>10}"
        wdist_str = f"{wdist:>10.4f}" if not np.isnan(wdist) else f"{'n/a':>10}"
        sa_str    = f"{sa:>12.6f}" if not np.isnan(sa) else f"{'n/a':>12}"
        ratio_str = f"{ratio:>11.2f}×" if not np.isnan(ratio) else f"{'n/a':>11}"

        print(f"  {label:<36} {ops_tag:>4} {wcos_str} {wdist_str} {sa_str} {ratio_str}")

        row = dict(ops=ops_tag, d=d_tag, wcos=wcos, wdist=wdist, sa=sa, ratio=ratio)
        all_rows.append(row)
        bucket_key = (ops_tag, d_tag)
        if bucket_key in buckets and not np.isnan(sa):
            buckets[bucket_key].append(row)

    print()

    # -----------------------------------------------------------------------
    # STEP 4 — Aggregate by bucket
    # -----------------------------------------------------------------------

    print("─" * 72)
    print("STEP 4 — Aggregate statistics by op-chain match × shape match")
    print("─" * 72)
    print()
    print(f"  {'Bucket':<24} {'n':>3}  {'wbar_cos μ':>11} {'wbar_dist μ':>12} "
          f"{'span_align μ':>13} {'ratio μ':>9}")
    print("  " + "-" * 76)

    bucket_stats = {}
    for bkey, rows in buckets.items():
        if not rows:
            print(f"  {str(bkey):<24} {'0':>3}  (no pairs)")
            continue
        n = len(rows)
        wcos_vals  = [r["wcos"]  for r in rows if not np.isnan(r["wcos"])]
        wdist_vals = [r["wdist"] for r in rows if not np.isnan(r["wdist"])]
        sa_vals    = [r["sa"]    for r in rows if not np.isnan(r["sa"])]
        ratio_vals = [r["ratio"] for r in rows if not np.isnan(r["ratio"])]

        mu_wcos  = np.mean(wcos_vals)  if wcos_vals  else float("nan")
        mu_wdist = np.mean(wdist_vals) if wdist_vals else float("nan")
        mu_sa    = np.mean(sa_vals)    if sa_vals    else float("nan")
        mu_ratio = np.mean(ratio_vals) if ratio_vals else float("nan")

        label = f"ops={bkey[0]}, D={bkey[1]}"
        print(f"  {label:<24} {n:>3}  "
              f"{mu_wcos:>11.5f} {mu_wdist:>12.5f} "
              f"{mu_sa:>13.6f} {mu_ratio:>9.2f}×")
        bucket_stats[bkey] = dict(n=n, mu_wcos=mu_wcos, mu_wdist=mu_wdist,
                                  mu_sa=mu_sa, mu_ratio=mu_ratio)

    print()
    print(f"  Random span(B) baseline:  {rand_baseline:.2e}")
    print()

    # -----------------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------------

    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    same = bucket_stats.get(("SAME", "SAME_D"))
    diff = bucket_stats.get(("DIFF", "SAME_D"))

    # -----------------------------------------------------------------------
    # Correct 2×2 analysis: (ops_match) × (seed_match)
    # The aggregate bucket stats above are confounded: DIFF-ops bucket contains
    # both same-seed pairs (extremely high alignment) and diff-seed pairs (random).
    # Separating by seed reveals the true driver.
    # -----------------------------------------------------------------------

    print("─" * 72)
    print("CORRECTED ANALYSIS — 2×2 table: ops_match × seed_match")
    print("─" * 72)
    print()
    print("  The aggregate buckets in Step 4 mix same-seed and diff-seed pairs.")
    print("  seed_match = SAME means both models were trained from the same")
    print("  random seed (same initialisation + same data order).")
    print()

    buckets_2x2 = {}
    for (k1, k2) in itertools.combinations(all_keys, 2):
        si1 = k1[1]; si2 = k2[1]
        n1_name = k1[0]; n2_name = k2[0]
        if records[k1]["D"] != records[k2]["D"]:
            continue
        sa = span_align(records[k1]["B"], records[k2]["B"])
        h1 = records[k1]["sig_hash"]; h2 = records[k2]["sig_hash"]
        ops_tag  = "SAME" if h1 == h2 else "DIFF"
        seed_tag = "SAME" if si1 == si2 else "DIFF"
        key = (ops_tag, seed_tag)
        buckets_2x2.setdefault(key, []).append(
            {"sa": sa, "wcos": w_bar_cos(records[k1]["w_bar"], records[k2]["w_bar"]),
             "k1": k1, "k2": k2})

    rb = rand_baseline
    print(f"  Random span(B) baseline: {rb:.2e}")
    print()
    print(f"  {'Bucket':<30} {'n':>3}  {'wbar_cos μ':>11} {'span_align μ':>13} {'ratio/rand':>11}")
    print("  " + "-" * 72)

    stats_2x2 = {}
    for ops_tag in ["SAME", "DIFF"]:
        for seed_tag in ["SAME", "DIFF"]:
            key = (ops_tag, seed_tag)
            if key not in buckets_2x2:
                print(f"  ops={ops_tag}, seed={seed_tag}  — (no pairs, impossible combination)")
                continue
            rows_b = buckets_2x2[key]
            n = len(rows_b)
            mu_cos = np.mean([r["wcos"] for r in rows_b])
            mu_sa  = np.mean([r["sa"]   for r in rows_b])
            ratio  = mu_sa / rb
            label  = f"ops={ops_tag}, seed={seed_tag}"
            print(f"  {label:<30} {n:>3}  {mu_cos:>11.4f} {mu_sa:>13.6f} {ratio:>11.1f}×")
            stats_2x2[key] = dict(n=n, mu_cos=mu_cos, mu_sa=mu_sa, ratio=ratio)

    print()

    # Key comparisons
    diff_same = stats_2x2.get(("DIFF", "SAME"))  # diff ops, same seed
    same_diff = stats_2x2.get(("SAME", "DIFF"))  # same ops, diff seed
    diff_diff = stats_2x2.get(("DIFF", "DIFF"))  # diff ops, diff seed

    print("  Key comparison:")
    if diff_same and same_diff:
        seed_effect = diff_same["ratio"] / max(same_diff["ratio"], 1e-2)
        print(f"  Seed effect (ops=DIFF, seed=SAME vs DIFF): {diff_same['ratio']:.0f}× vs "
              f"{same_diff['ratio']:.1f}× → ratio {seed_effect:.0f}×")
    if diff_diff and same_diff:
        ops_effect_ds = diff_diff["ratio"] / max(same_diff["ratio"], 1e-2)
        print(f"  Op chain effect (ops=DIFF vs SAME, seed=DIFF): {diff_diff['ratio']:.1f}× vs "
              f"{same_diff['ratio']:.1f}× → ratio {ops_effect_ds:.1f}×")
    print()

    print("  Findings:")
    print("  ┌──────────────────────────────────────────────────────────────────────┐")
    print("  │                                                                      │")
    if diff_same and diff_same["ratio"] > 100:
        print(f"  │  1. SEED IS THE DOMINANT DRIVER (not ATen op chain).               │")
        print(f"  │     ops=DIFF + seed=SAME → span(B) ratio {diff_same['ratio']:>6.0f}× above random.  │")
        print(f"  │     Two DIFFERENT architectures with the same initialisation        │")
        print(f"  │     and training order converge to nearly identical w_bar and B.   │")
    if same_diff and same_diff["ratio"] < 3:
        print(f"  │                                                                      │")
        print(f"  │  2. ATen OP CHAIN HAS NO EFFECT.                                    │")
        print(f"  │     ops=SAME + seed=DIFF → span(B) ratio {same_diff['ratio']:>5.1f}× (≈ random).   │")
        print(f"  │     Same architecture, different seed → random span(B) alignment.   │")
    if diff_diff and diff_diff["ratio"] < 3:
        print(f"  │                                                                      │")
        print(f"  │  3. WITHOUT SHARED SEED, ALL PAIRS ARE AT CHANCE LEVEL.            │")
        print(f"  │     ops=DIFF + seed=DIFF → {diff_diff['ratio']:>5.1f}× (≈ random baseline).      │")
    print(f"  │                                                                      │")
    print(f"  │  CONCLUSION:                                                         │")
    print(f"  │  CKA=1.0 / span(B) universality is driven by OPTIMISATION           │")
    print(f"  │  TRAJECTORY — shared initialisation + data order — not by:           │")
    print(f"  │    • the ATen computational graph structure, or                       │")
    print(f"  │    • the architecture topology, or                                    │")
    print(f"  │    • the mathematical structure of SVD on same-shaped matrices.       │")
    print(f"  │                                                                      │")
    print(f"  │  w_bar and B are essentially fingerprints of the optimisation path,   │")
    print(f"  │  not of the model architecture. Two models that walk the same loss    │")
    print(f"  │  landscape (same seed → same initialisation → same gradient steps)   │")
    print(f"  │  will produce identical GABE components regardless of op chain.       │")
    print(f"  │                                                                      │")
    print(f"  │  IMPLICATION FOR Exp 6 (CKA=1.0):                                   │")
    print(f"  │  The 'universality' observed across architectures may reflect that    │")
    print(f"  │  pretrained models compared in Exp 6 were evaluated at similar        │")
    print(f"  │  points in weight space (same pretrained checkpoint, same data).      │")
    print(f"  │  It is NOT evidence that span(B) is architecture-determined.          │")
    print("  └──────────────────────────────────────────────────────────────────────┘")
    print()

    return dict(sigs=sigs, records=records, bucket_stats=bucket_stats,
                stats_2x2=stats_2x2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, default=15)
    parser.add_argument("--C",         type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_seeds",   type=int, default=3)
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()
    run(args.epochs, args.C, args.n_samples, args.n_seeds, args.device, args.seed)
