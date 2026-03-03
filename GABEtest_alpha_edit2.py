# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_alpha_edit2.py — Experiment 20b: α-Editing with Relative Noise Normalization
#
# PURPOSE:
#   Corrected version of Experiment 20. Two fixes over the original:
#
#   FIX 1 — Metric: consistency vs baseline, not ground-truth accuracy.
#     ResNet-18 has 1000 ImageNet outputs; CIFAR-10 labels (0-9) are irrelevant.
#     We measure: what fraction of predictions change after an edit?
#     Baseline consistency = 1.0 by definition.
#
#   FIX 2 — Fair noise comparison via relative (ε-norm) scaling.
#     Original bug: noise_wbar used coeffs.std() as scale, but α and W_bar
#     live in different spaces with different magnitudes. This made the
#     comparison meaningless (and caused KL=inf for W_bar).
#
#     Correct approach: for a given ε, add noise of magnitude ε·‖component‖_F
#     to each component independently. Same relative perturbation → fair comparison.
#
#       noise_α    = randn_like(α)    / ‖randn‖_F  × ε × ‖α‖_F
#       noise_Wbar = randn_like(Wbar) / ‖randn‖_F  × ε × ‖Wbar_residuals‖_F
#
#     We sweep ε ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5} and record
#     KL divergence and consistency at each level for both components.
#
#   OUTPUT:
#     - Table of structural edits (ZERO, SCALE, SWAP, INTERP, SHUFFLE)
#     - Noise sweep table: KL and Consist. at each ε for α vs W_bar
#     - ε_50: the noise level where consistency first drops below 0.50
#     - Sensitivity ratio at each ε: KL_α / KL_Wbar
#
# USAGE:
#   python GABEtest_alpha_edit2.py
#   python GABEtest_alpha_edit2.py --shape 64 64 3 3 --n_eval 256

import sys, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import copy

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_group_modules(model, target_shape):
    return [m for m in model.modules()
            if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == target_shape]


def cache_predictions(model, loader, device, max_batches=20):
    """Cache baseline top-1 predictions. Ground-truth labels are never used."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches: break
            preds.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(preds)


def evaluate_consistency(model_edit, loader, baseline_preds, device, max_batches=20):
    """Fraction of inputs where edited model top-1 matches baseline. 1.0 = no change."""
    model_edit.eval()
    agree, total = 0, 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches: break
            pred = model_edit(x.to(device)).argmax(1).cpu()
            agree += (pred == baseline_preds[total: total + len(pred)]).sum().item()
            total += len(pred)
    return agree / max(total, 1)


def output_divergence(model_ref, model_mod, loader, device, max_batches=10):
    """Mean KL(baseline ‖ edited). Clamped to avoid log(0)=−inf."""
    model_ref.eval(); model_mod.eval()
    kl_vals = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches: break
            x = x.to(device)
            p = torch.softmax(model_ref(x), dim=-1).clamp(1e-10)
            q = torch.softmax(model_mod(x), dim=-1).clamp(1e-10)
            kl = (p * (p.log() - q.log())).sum(-1).mean().item()
            kl_vals.append(kl)
    return float(np.mean(kl_vals))


def apply_gabe_edit(model_orig, target_shape, edit_fn, device):
    """Compress → edit via edit_fn → decompress → write back. Returns new model copy."""
    model = copy.deepcopy(model_orig)
    gabe = GABE()
    mods = get_group_modules(model, target_shape)
    ws = [m.weight.detach().clone() for m in mods]
    compressed = gabe.compress(ws, basis_rank=1, w_bar_rank=16)
    compressed = edit_fn(compressed)
    recon = gabe.decompress(compressed)
    for m, w_new in zip(mods, recon):
        m.weight.data.copy_(w_new.view(m.weight.shape))
    return model


def relative_noise(tensor, eps, generator=None):
    """
    Returns noise of Frobenius norm = eps * ‖tensor‖_F.
    Direction is random unit vector in the same shape as tensor.
    This gives a fair same-relative-magnitude perturbation regardless of tensor scale.
    """
    raw = torch.randn(tensor.shape, generator=generator)
    raw_norm = raw.norm()
    if raw_norm < 1e-12:
        return torch.zeros_like(tensor)
    return raw / raw_norm * eps * tensor.norm()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(target_shape=(64, 64, 3, 3), n_eval=256, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = torch.Generator(); rng.manual_seed(seed)

    print("=" * 66)
    print("GABE Experiment 20b: α-Editing with Relative Noise Normalization")
    print("=" * 66)
    print(f"shape={target_shape}  n_eval={n_eval}")
    print()

    tf = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, list(range(n_eval))),
        batch_size=32, shuffle=False)

    print("[1] Loading pretrained ResNet-18...")
    model_base = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(device)
    mods = get_group_modules(model_base, target_shape)
    if len(mods) < 2:
        print("ERROR: need >= 2 layers with this shape. Available shapes:")
        for s in sorted({tuple(m.weight.shape) for m in model_base.modules()
                         if isinstance(m, nn.Conv2d)}):
            print(f"  {s}")
        return

    gabe = GABE()
    ws_base = [m.weight.detach().clone() for m in mods]
    compressed_base = gabe.compress(ws_base, basis_rank=1, w_bar_rank=16)
    coeffs_base = compressed_base["coeffs"].clone()     # (L, K)
    wbar_res    = compressed_base["w_bar_residuals"]    # shape of w_bar
    L, K = coeffs_base.shape

    # Norms of the two components being compared
    norm_alpha = coeffs_base.norm().item()
    norm_wbar  = wbar_res.norm().item()

    baseline_preds = cache_predictions(model_base, loader, device)
    print(f"  Group: {L} layers, K={K}")
    print(f"  ‖α‖_F        = {norm_alpha:.4f}  (shape {list(coeffs_base.shape)})")
    print(f"  ‖W_bar_res‖_F = {norm_wbar:.4f}  (shape {list(wbar_res.shape)})")
    print()

    # ------------------------------------------------------------------
    # PART 1: Structural edits (same as Experiment 20, for comparison)
    # ------------------------------------------------------------------

    print("─" * 66)
    print("PART 1 — Structural edits")
    print("─" * 66)
    print(f"  {'Edit':<32} {'Consist.':>10} {'KL_div':>10} {'ConsΔ':>8}")
    print("  " + "-" * 64)

    def run_edit(name, edit_fn):
        m_edit = apply_gabe_edit(model_base, target_shape, edit_fn, device)
        cons = evaluate_consistency(m_edit, loader, baseline_preds, device)
        kl   = output_divergence(model_base, m_edit, loader, device)
        print(f"  {name:<32} {cons:>10.4f} {kl:>10.4f} {cons-1.0:>+8.4f}")
        return dict(cons=cons, kl=kl)

    struct_results = {}
    struct_results["ZERO"]     = run_edit("ZERO  α→0",
        lambda c: {**c, "coeffs": torch.zeros_like(c["coeffs"])})
    struct_results["SCALE_2x"] = run_edit("SCALE  α→2α",
        lambda c: {**c, "coeffs": c["coeffs"] * 2.0})
    struct_results["SCALE_05"] = run_edit("SCALE  α→0.5α",
        lambda c: {**c, "coeffs": c["coeffs"] * 0.5})

    def swap_edit(c):
        nc = c["coeffs"].clone()
        nc[0], nc[-1] = c["coeffs"][-1].clone(), c["coeffs"][0].clone()
        return {**c, "coeffs": nc}
    struct_results["SWAP"]   = run_edit("SWAP  α[0]↔α[-1]", swap_edit)

    def interp_edit(c, t=0.5):
        nc = c["coeffs"].clone()
        nc[0] = t * c["coeffs"][0] + (1-t) * c["coeffs"][1]
        nc[1] = t * c["coeffs"][1] + (1-t) * c["coeffs"][0]
        return {**c, "coeffs": nc}
    struct_results["INTERP"] = run_edit("INTERP  α[0]↔α[1] t=0.5", interp_edit)

    def shuffle_edit(c):
        idx = torch.randperm(L, generator=rng)
        return {**c, "coeffs": c["coeffs"][idx]}
    struct_results["SHUFFLE"] = run_edit("SHUFFLE  α random permute", shuffle_edit)

    # ------------------------------------------------------------------
    # PART 2: Relative noise sweep — fair comparison α vs W_bar
    # ------------------------------------------------------------------

    print()
    print("─" * 66)
    print("PART 2 — Relative noise sweep  (noise = ε × ‖component‖_F)")
    print("─" * 66)
    print(f"  {'ε':>7}  "
          f"{'α Cons.':>9} {'α KL':>8}  "
          f"{'Wbar Cons.':>11} {'Wbar KL':>9}  "
          f"{'KL ratio α/W':>13}")
    print("  " + "-" * 64)

    eps_levels = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    sweep = []

    for eps in eps_levels:
        # Noise on α — relative to ‖α‖_F
        def noise_alpha(c, _eps=eps):
            n = relative_noise(c["coeffs"], _eps)
            return {**c, "coeffs": c["coeffs"] + n.to(c["coeffs"].dtype)}

        # Noise on W_bar residuals — relative to ‖W_bar_res‖_F
        def noise_wbar(c, _eps=eps):
            n = relative_noise(c["w_bar_residuals"], _eps)
            return {**c, "w_bar_residuals": c["w_bar_residuals"] + n.to(c["w_bar_residuals"].dtype)}

        m_a = apply_gabe_edit(model_base, target_shape, noise_alpha, device)
        m_w = apply_gabe_edit(model_base, target_shape, noise_wbar,  device)

        cons_a = evaluate_consistency(m_a, loader, baseline_preds, device)
        cons_w = evaluate_consistency(m_w, loader, baseline_preds, device)
        kl_a   = output_divergence(model_base, m_a, loader, device)
        kl_w   = output_divergence(model_base, m_w, loader, device)

        ratio_str = f"{kl_a/kl_w:>13.3f}" if kl_w > 1e-8 else "         N/A"
        print(f"  {eps:>7.3f}  "
              f"{cons_a:>9.4f} {kl_a:>8.4f}  "
              f"{cons_w:>11.4f} {kl_w:>9.4f}  "
              f"{ratio_str}")

        sweep.append(dict(eps=eps, cons_a=cons_a, kl_a=kl_a, cons_w=cons_w, kl_w=kl_w))

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------

    print()
    print("=" * 66)
    print("ANALYSIS")
    print("=" * 66)

    # ε_50: first ε where consistency drops below 0.50
    def find_eps50(key_cons):
        for row in sweep:
            if row[key_cons] < 0.50:
                return row["eps"]
        return None

    eps50_a = find_eps50("cons_a")
    eps50_w = find_eps50("cons_w")

    def fmt_eps50(v):
        return f"{v:.3f}" if v is not None else ">1.0 (never)"

    print(f"  ε_50 for α          : {fmt_eps50(eps50_a)}")
    print(f"  ε_50 for W_bar_res  : {fmt_eps50(eps50_w)}")
    print()

    if eps50_a is not None and eps50_w is not None:
        if eps50_w < eps50_a:
            ratio50 = eps50_a / eps50_w
            print(f"  W_bar_res breaks first (ratio {ratio50:.1f}×). α is MORE robust.")
            print(f"  → Supports α-as-pointer: small α changes have outsized effect,")
            print(f"    but W_bar_res is even more fragile.")
        elif eps50_a < eps50_w:
            ratio50 = eps50_w / eps50_a
            print(f"  α breaks first (ratio {ratio50:.1f}×). α is MORE sensitive than W_bar_res.")
            print(f"  → Consistent with pointer hypothesis: α is a compact, high-leverage")
            print(f"    control surface; small perturbations change behavior more than")
            print(f"    equivalent relative noise on the weight basis.")
        else:
            print(f"  α and W_bar_res break at the same ε. Similar sensitivity.")
    elif eps50_a is None and eps50_w is None:
        print("  Neither component dropped below 0.50 consistency in tested ε range.")
        print("  Try larger ε values.")
    elif eps50_a is None:
        print(f"  α never drops below 0.50 — very robust to noise.")
        print(f"  W_bar_res breaks at ε={fmt_eps50(eps50_w)}.")
    else:
        print(f"  W_bar_res never drops below 0.50 — very robust to noise.")
        print(f"  α breaks at ε={fmt_eps50(eps50_a)}.")

    print()

    # KL ratio at small ε (most informative; linear regime before saturation)
    small_eps_rows = [r for r in sweep if r["eps"] <= 0.05 and r["kl_w"] > 1e-8]
    if small_eps_rows:
        ratios = [r["kl_a"] / r["kl_w"] for r in small_eps_rows]
        avg_ratio = np.mean(ratios)
        print(f"  Mean KL ratio α/W_bar (ε ≤ 0.05) : {avg_ratio:.3f}×")
        if avg_ratio > 2.0:
            print("  → α is MORE sensitive per unit relative perturbation")
            print("    (supports pointer hypothesis: α is a high-leverage control vector)")
        elif avg_ratio < 0.5:
            print("  → α is LESS sensitive per unit relative perturbation")
            print("    (W_bar_res carries more behavioral information at small scales)")
        else:
            print("  → α and W_bar_res have comparable sensitivity (ratio near 1×)")

    print()
    zero_cons = struct_results["ZERO"]["cons"]
    print(f"  ZERO α consistency : {zero_cons:.4f}  (drop = {zero_cons - 1.0:+.4f})")
    if zero_cons < 0.50:
        print("  → α carries substantial behavioral information.")
        print("    Collapsing to W_bar alone changes >50% of predictions.")
    else:
        print("  → W_bar alone preserves most predictions. α has limited leverage here.")

    return dict(struct=struct_results, sweep=sweep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",  type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--n_eval", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.n_eval, args.device, args.seed)
