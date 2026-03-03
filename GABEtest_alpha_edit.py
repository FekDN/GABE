# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_alpha_edit.py — Experiment 20: α-Editing Functional Test
#
# PURPOSE:
#   Tests whether editing the GABE coefficients α_i produces predictable,
#   targeted behavioral changes in the model — consistent with the "pointer"
#   interpretation where α selects which behavior is active.
#
#   Four editing operations:
#     ZERO:        α → 0  for all layers (collapse all layers to W_bar)
#     SCALE_UP:    α → α × s  (amplify current addressing)
#     SWAP(i,j):   α_i ↔ α_j  (swap layer behaviors)
#     INTERPOLATE: α_new = t·α_A + (1-t)·α_B  (blend two layers)
#
#   Each operation is applied, the modified model is evaluated,
#   and the output divergence from baseline is measured.
#
#   EXPECTED RESULT:
#     - ZERO: consistency drop (predictions change when collapsed to W_bar)
#     - SCALE_UP: varies; may preserve or disrupt predictions depending on scale
#     - SWAP(i,j): local behavior exchange; affects predictions like swapping layers
#     - INTERPOLATE: smooth gradient between i and j behaviors
#
#   NOTE: Model is ResNet-18 pretrained on ImageNet (1000 classes). CIFAR-10 ground-truth
#   labels are intentionally ignored. Metrics are behavioral: consistency (% of inputs
#   where edited model top-1 agrees with baseline) and KL divergence of output distributions.
#
#   COMPARISON: same operations applied to W_bar residuals → should be more robust.
#
# USAGE:
#   python GABEtest_alpha_edit.py
#   python GABEtest_alpha_edit.py --shape 64 64 3 3 --n_eval 256

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
    """Cache baseline argmax predictions. Labels are irrelevant -- model has 1000 ImageNet outputs."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches: break
            preds.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(preds)   # (N,)


def evaluate_consistency(model_edit, loader, baseline_preds, device, max_batches=20):
    """Fraction of inputs where edited model top-1 prediction matches baseline.
    Baseline consistency = 1.0 by definition. Drops indicate behavioral change from the edit."""
    model_edit.eval()
    agree, total = 0, 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches: break
            x = x.to(device)
            pred = model_edit(x).argmax(1).cpu()
            agree += (pred == baseline_preds[total: total + len(pred)]).sum().item()
            total += len(pred)
    return agree / max(total, 1)


def output_divergence(model_ref, model_mod, loader, device, max_batches=10):
    """Mean KL divergence of output distributions between two models."""
    model_ref.eval(); model_mod.eval()
    kl_vals = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches: break
            x = x.to(device)
            p = torch.softmax(model_ref(x), dim=-1)
            q = torch.softmax(model_mod(x), dim=-1)
            kl = (p * (p.log() - q.log())).sum(-1).mean().item()
            kl_vals.append(kl)
    return np.mean(kl_vals)


def apply_gabe_edit(model_orig, target_shape, edit_fn, device):
    """
    Applies a GABE-based edit:
      1. Compress all layers in the group.
      2. Call edit_fn(compressed_data) to modify components.
      3. Decompress and write back.
    Returns a modified model copy.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(target_shape=(64, 64, 3, 3), n_eval=256, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 20: α-Editing Functional Test")
    print("=" * 62)
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
        print(f"ERROR: need ≥2 layers. Available shapes:")
        for s in sorted({tuple(m.weight.shape) for m in model_base.modules()
                         if isinstance(m, nn.Conv2d)}):
            print(f"  {s}")
        return

    # Get baseline coefficients for reference
    gabe = GABE()
    ws_base = [m.weight.detach().clone() for m in mods]
    compressed_base = gabe.compress(ws_base, basis_rank=1, w_bar_rank=16)
    coeffs_base = compressed_base["coeffs"].clone()   # (L, K)
    L, K = coeffs_base.shape
    coeff_norm = coeffs_base.norm(dim=1)   # per-layer coefficient norm

    # Cache baseline predictions once (ResNet-18 has 1000 ImageNet outputs;
    # CIFAR-10 ground-truth labels are irrelevant -- we measure behavioral consistency)
    baseline_preds = cache_predictions(model_base, loader, device)
    print(f"  Baseline: {L} layers, K={K}  (consistency vs self = 1.0000 by definition)")
    print(f"  Coeff norms per layer: {coeff_norm.tolist()}")

    print()
    print(f"  {'Edit':<28} {'Consist.':>10} {'KL_div':>10} {'ConsΔ':>8}")
    print("  " + "-" * 62)

    def run_edit(name, edit_fn):
        m_edit = apply_gabe_edit(model_base, target_shape, edit_fn, device)
        cons = evaluate_consistency(m_edit, loader, baseline_preds, device)
        kl   = output_divergence(model_base, m_edit, loader, device)
        delta = cons - 1.0
        print(f"  {name:<28} {cons:>10.4f} {kl:>10.4f} {delta:>+8.4f}")
        return dict(cons=cons, kl=kl, delta=delta)

    results = {}

    # ZERO: set all coefficients to 0
    results["ZERO_alpha"] = run_edit(
        "ZERO  α→0",
        lambda c: {**c, "coeffs": torch.zeros_like(c["coeffs"])})

    # SCALE_UP × 2
    results["SCALE_2x"] = run_edit(
        "SCALE  α→2α",
        lambda c: {**c, "coeffs": c["coeffs"] * 2.0})

    # SCALE_DOWN × 0.5
    results["SCALE_05x"] = run_edit(
        "SCALE  α→0.5α",
        lambda c: {**c, "coeffs": c["coeffs"] * 0.5})

    # SWAP first and last layers' coefficients
    def swap_edit(c):
        nc = c["coeffs"].clone()
        nc[0], nc[-1] = c["coeffs"][-1].clone(), c["coeffs"][0].clone()
        return {**c, "coeffs": nc}
    results["SWAP_0_last"] = run_edit("SWAP  α[0]↔α[-1]", swap_edit)

    # INTERPOLATE: blend α[0] and α[1]
    def interp_edit(c, t=0.5):
        nc = c["coeffs"].clone()
        nc[0] = t * c["coeffs"][0] + (1-t) * c["coeffs"][1]
        nc[1] = t * c["coeffs"][1] + (1-t) * c["coeffs"][0]
        return {**c, "coeffs": nc}
    results["INTERP_0_1"] = run_edit("INTERP  α[0]↔α[1] (t=0.5)", interp_edit)

    # SHUFFLE: randomly permute all coefficients
    def shuffle_edit(c):
        idx = torch.randperm(L)
        return {**c, "coeffs": c["coeffs"][idx]}
    results["SHUFFLE"] = run_edit("SHUFFLE  α random permute", shuffle_edit)

    # NOISE_ALPHA: add Gaussian noise to α only
    def noise_alpha_edit(c, scale=1.0):
        noise = torch.randn_like(c["coeffs"]) * scale * c["coeffs"].std()
        return {**c, "coeffs": c["coeffs"] + noise}
    results["NOISE_ALPHA_1x"] = run_edit("NOISE_α  σ=1×std", noise_alpha_edit)

    # NOISE_WBAR: same noise magnitude but on W_bar residuals (control)
    def noise_wbar_edit(c, scale=1.0):
        noise = torch.randn_like(c["w_bar_residuals"]) * scale * c["coeffs"].std()
        return {**c, "w_bar_residuals": c["w_bar_residuals"] + noise.to(c["w_bar_residuals"].dtype)}
    results["NOISE_Wbar_1x"] = run_edit("NOISE_Wbar  σ=1×coeff_std", noise_wbar_edit)

    print()
    print("=" * 62)
    print("ANALYSIS")
    print("=" * 62)
    kl_alpha = results["NOISE_ALPHA_1x"]["kl"]
    kl_wbar  = results["NOISE_Wbar_1x"]["kl"]
    print(f"KL divergence — NOISE_α    : {kl_alpha:.6f}")
    print(f"KL divergence — NOISE_Wbar : {kl_wbar:.6f}")
    if kl_wbar > 1e-8:
        ratio = kl_alpha / kl_wbar
        print(f"Sensitivity ratio α/Wbar   : {ratio:.2f}×")
        if ratio > 2:
            print("→ α is MORE sensitive than W_bar (supports pointer hypothesis)")
        elif ratio < 0.5:
            print("→ α is LESS sensitive (contradicts pointer hypothesis)")
        else:
            print("→ α and W_bar have similar sensitivity at this noise scale")
    print()
    zero_delta = results["ZERO_alpha"]["delta"]
    print(f"ZERO α consistency drop    : {zero_delta:+.4f}")
    if zero_delta < -0.05:
        print("→ Coefficients carry behavioral information (predictions change without them)")
    else:
        print("→ W_bar alone is nearly sufficient (coefficients not critical at this scale)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",  type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--n_eval", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.n_eval, args.device, args.seed)
