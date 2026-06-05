# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_tucker.py  —  Experiment 31: Tucker-GABE — Rayleigh Alignment in Kernel Space
#
# ════════════════════════════════════════════════════════════════════════════
# MOTIVATION
# ════════════════════════════════════════════════════════════════════════════
#
# The GABE decomposition:
#     W_i = W_bar + sum_k alpha_i[k] * B_k
# operates in the *original* flattened weight space R^D.
#
# Two problems with large D:
#   (A) CROSS-SHAPE GROUPING: layers (64,64,3,3) and (128,128,3,3) have
#       D=36864 and D=147456 — impossible to GABE together.
#   (B) SCALABILITY: Fisher MVP costs O(N × D); for D=2M this is slow.
#
# Tucker decomposition offers a solution:
#   W ≈ G ×_1 U1 ×_2 U2 ×_3 U3 ×_4 U4
#   Core G ∈ R^{r_out × r_in × kH × kW}  (same size for ALL groups at fixed r)
#   Factor matrices Ui are orthonormal bases for each mode.
#
# Tucker-GABE pipeline:
#   1. Compute Tucker factors from the group (HOSVD on stacked layers)
#   2. Project each W_i to core G_i ∈ R^{r × r × kH × kW}
#   3. Apply standard GABE to {G_i} in the (smaller) kernel space
#   4. B_k_core live in R^{r²·kH·kW} — same size for ALL groups at fixed r
#   5. Cross-group GABE becomes possible!
#
# ════════════════════════════════════════════════════════════════════════════
# CRITICAL QUESTION: Rayleigh alignment preservation
# ════════════════════════════════════════════════════════════════════════════
#
# Experiments 8-12 established:
#   B_1, B_2 in R^D → 99-100th percentile of Rayleigh spectrum (H, F, GCM)
#   B_3 → ~35th percentile
#
# After Tucker projection to R^{r²·kH·kW}:
#   Do B_k_core still sit at the 99th+ percentile of the KERNEL-SPACE Fisher?
#
# If YES → entire chain of evidence from Exp 8-12 transfers to Tucker-GABE.
#           Tucker is a lossless(ish) structural compression.
# If NO  → Tucker projection destroys precisely the property we measured.
#           Kernel-space GABE is structurally different from flat-space GABE.
#
# ════════════════════════════════════════════════════════════════════════════
# THEORETICAL BACKGROUND
# ════════════════════════════════════════════════════════════════════════════
#
# The Tucker reconstruction map  φ: G → W is linear:
#   φ(G) = G ×_1 U1 ×_2 U2 ×_3 U3 ×_4 U4
#   Jacobian: J = U4 ⊗ U3 ⊗ U2 ⊗ U1   (Kronecker product)
#   Shape: (D, d_core) where D = C_out*C_in*kH*kW, d_core = r²·kH·kW
#
# Fisher in kernel space:
#   F_core = J^T F J
#
# Kernel-space Fisher MVP (efficient, no full-space matrix needed):
#   F_core(v) = J^T [ F (J v) ]
#   Step 1: Jv = tucker_reconstruct(v_core, factors)  → R^D
#   Step 2: F(Jv) = Fisher MVP in full space with direction Jv
#   Step 3: J^T [F(Jv)] = tucker_project(F(Jv), factors) → R^d_core
#
# Rayleigh quotient in kernel space:
#   λ_core(v) = v^T F_core v / ||v||²
#              = (Jv)^T F (Jv) / ||v||²
#
# Preservation condition: if B_k lies entirely in range(J), the Rayleigh
# quotient is preserved up to scaling by ||J^T B_k||² / ||B_k||².
# At low rank r, B_k has components in range(J)^⊥ that are lost.
#
# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PARTS
# ════════════════════════════════════════════════════════════════════════════
#
# Part A — Tucker quality sweep: r ∈ {2, 4, 8, 16, 32}
#   Measures: reconstruction RMSE, variance explained, compression ratio.
#   Same groups as Exp 22 (ResNet-18 layer groups).
#
# Part B — Rayleigh alignment preservation test
#   For each group × r combination:
#     (1) Original B_k Rayleigh percentile (in R^D, using flat Fisher)
#     (2) Core B_k Rayleigh percentile (in R^d_core, using kernel Fisher)
#     (3) Preservation ratio: core_pct / original_pct
#     (4) At what r is preservation > 90%?
#
# Part C — Cross-shape Tucker-GABE
#   Take ALL ResNet-18 groups simultaneously.
#   Project each to same kernel R^{r × r × kH × kW}.
#   Apply GABE across ALL layers from ALL groups.
#   Measure: does cross-shape GABE find meaningful B_k?
#   Rayleigh percentile of cross-shape B_k in kernel Fisher.
#
# Part D — Alignment budget in kernel space
#   For each r: how much of B_k's original Rayleigh quotient survives?
#   Plot: preservation% vs r, with theoretical bound ||J B_k_comp||/||B_k||.
#
# ════════════════════════════════════════════════════════════════════════════
# USAGE
# ════════════════════════════════════════════════════════════════════════════
#
#   python GABEtest_tucker.py
#   python GABEtest_tucker.py --ranks "2,4,8,16,32" --n_grad 64
#   python GABEtest_tucker.py --device cuda --ranks "4,8,16,32,64"
#   python GABEtest_tucker.py --parts AB   (skip cross-shape grouping)

import sys, os, argparse, time
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision
import torchvision.transforms as transforms
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ─────────────────────────────────────────────────────────────────────────────
# Tucker decomposition (HOSVD — Higher-Order SVD)
# ─────────────────────────────────────────────────────────────────────────────

def tensor_unfold(tensor, mode):
    """
    Mode-n unfolding (matricization) of a tensor.
    Returns: matrix of shape (tensor.shape[mode], prod(other_dims))
    """
    n = tensor.dim()
    order = [mode] + [i for i in range(n) if i != mode]
    return tensor.permute(order).reshape(tensor.shape[mode], -1)


def tensor_mode_product(tensor, matrix, mode):
    """
    Mode-n product: multiply tensor by matrix along dimension `mode`.
    tensor shape: (..., n, ...)
    matrix shape: (r, n)
    result shape: (..., r, ...)
    """
    # Move target mode to front, reshape, multiply, reshape back
    n = tensor.dim()
    order = [mode] + [i for i in range(n) if i != mode]
    t = tensor.permute(order)                       # (n, *other)
    shape_rest = t.shape[1:]
    t = t.reshape(tensor.shape[mode], -1)           # (n, prod_other)
    t = matrix @ t                                  # (r, prod_other)
    t = t.reshape((matrix.shape[0],) + shape_rest)  # (r, *other)
    inv_order = [0] * n
    for new_pos, old_pos in enumerate(order):
        inv_order[old_pos] = new_pos
    return t.permute(inv_order)


def hosvd(tensor, ranks):
    """
    Higher-Order SVD (Tucker-1 / sequentially truncated HOSVD).
    For each mode i, compute the leading `ranks[i]` left singular vectors
    of the mode-i unfolding.

    Args:
        tensor : torch.Tensor of any shape
        ranks  : list of ints, one per mode; each <= tensor.shape[i]

    Returns:
        core    : core tensor of shape ranks
        factors : list of factor matrices U_i of shape (tensor.shape[i], ranks[i])
    """
    tensor = tensor.to(torch.float64)
    n = tensor.dim()
    ranks = [min(r, tensor.shape[i]) for i, r in enumerate(ranks)]
    factors = []
    for mode in range(n):
        W_mode = tensor_unfold(tensor, mode)         # (shape[i], prod_rest)
        U, _, _ = torch.linalg.svd(W_mode, full_matrices=False)
        factors.append(U[:, :ranks[mode]])           # (shape[i], r_i)

    # Core = tensor contracted with U_i^T along each mode
    core = tensor.clone()
    for mode, U in enumerate(factors):
        core = tensor_mode_product(core, U.T, mode)  # successively shrinks dims

    return core, factors


def tucker_reconstruct(core, factors):
    """Reconstruct full tensor from Tucker decomposition."""
    result = core.clone()
    for mode, U in enumerate(factors):
        result = tensor_mode_product(result, U, mode)
    return result


def tucker_project(tensor, factors):
    """Project a full-space tensor to kernel space (core space)."""
    result = tensor.to(torch.float64).clone()
    for mode, U in enumerate(factors):
        result = tensor_mode_product(result, U.T, mode)
    return result


def hosvd_group(weight_list, ranks):
    """
    Compute shared Tucker factors from a group of tensors.
    Strategy: compute factors from the GROUP MEAN (W_bar).
    All tensors are then projected to the same kernel using W_bar's factors.
    This ensures a common coordinate system for GABE in kernel space.

    Returns:
        factors : list of factor matrices (from HOSVD of W_bar)
        cores   : list of core tensors, one per weight
        w_bar_core : core of the mean weight
    """
    ws = torch.stack(weight_list).to(torch.float64)
    w_bar = ws.mean(dim=0)
    _, factors = hosvd(w_bar, ranks)
    cores = [tucker_project(w, factors) for w in weight_list]
    w_bar_core = tucker_project(w_bar, factors)
    return factors, cores, w_bar_core


# ─────────────────────────────────────────────────────────────────────────────
# GABE in kernel space
# ─────────────────────────────────────────────────────────────────────────────

def gabe_kernel(cores):
    """
    Apply GABE SVD decomposition to a list of core tensors.
    Each core: shape (r_out, r_in, kH, kW) — treated as a flat vector.

    Returns: (w_bar_core, B_flat_normalised, coeffs, d_core)
    """
    g = GABE()
    # Flatten each core to 1D
    flat_cores = [c.reshape(-1).float() for c in cores]
    w_bar, B, coeffs, shape = g._extract_svd_components(flat_cores)
    d_core = len(flat_cores[0])
    B_flat = B.view(B.shape[0], -1).to(torch.float64)
    B_flat = B_flat / B_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return w_bar, B_flat, coeffs, d_core


# ─────────────────────────────────────────────────────────────────────────────
# Fisher MVP — original space and kernel space
# ─────────────────────────────────────────────────────────────────────────────

def collect_gradients(model, target_layers, loader, device, n_grad=64):
    """
    Collect per-sample gradients for a group of layers (mean across group).
    Returns: grads (n_grad, D) in float64
    """
    loss_fn = nn.CrossEntropyLoss()
    model.eval().to(device)
    grads = []
    for i, (x, y) in enumerate(loader):
        if len(grads) >= n_grad:
            break
        x, y = x[:1].to(device), y[:1].to(device)
        model.zero_grad()
        loss_fn(model(x), y).backward()
        gs = [l.weight.grad.detach().cpu().reshape(-1).to(torch.float64)
              for l in target_layers if l.weight.grad is not None]
        if gs:
            grads.append(torch.stack(gs).mean(0))
    model.zero_grad()
    if not grads:
        return None
    return torch.stack(grads)  # (N, D)


def fisher_mvp_flat(v, grads):
    """Empirical Fisher MVP in flat R^D. O(N·D)."""
    scores = grads @ v          # (N,)
    return (grads.T @ scores) / grads.shape[0]  # (D,)


def kernel_fisher_mvp(v_core_flat, grads, factors, weight_shape):
    """
    Kernel-space Fisher MVP.
    v_core_flat : direction in R^d_core
    grads       : (N, D) per-sample gradients in original R^D
    factors     : Tucker factor matrices
    weight_shape: original tensor shape (for reshaping)

    F_core(v) = J^T [F (J v)]
      Step 1: Jv = tucker_reconstruct(v, factors)  (lift to R^D)
      Step 2: F(Jv) = fisher_mvp(Jv, grads)        (full-space Fisher)
      Step 3: J^T result = tucker_project(result, factors)  (project back)
    """
    core_shape = tuple(f.shape[1] for f in factors)
    v_core = v_core_flat.reshape(core_shape).to(torch.float64)

    # Step 1: Lift to full space
    v_full = tucker_reconstruct(v_core, factors).reshape(-1)  # (D,)

    # Step 2: Full-space Fisher MVP
    Fv_full = fisher_mvp_flat(v_full, grads)  # (D,)

    # Step 3: Project back to kernel
    Fv_shape = Fv_full.reshape(weight_shape).to(torch.float64)
    Fv_core = tucker_project(Fv_shape, factors).reshape(-1)  # (d_core,)

    return Fv_core


def rayleigh_percentile(v_flat, mvp_fn, n_rand=500, d_core=None):
    """
    Compute Rayleigh quotient of v_flat under mvp_fn, and its percentile
    in the empirical distribution over n_rand random unit vectors.

    mvp_fn(v) -> F @ v  (kernel-space or original-space)
    d_core    : dimension for random vectors (defaults to len(v_flat))
    """
    d = d_core if d_core is not None else len(v_flat)
    v = v_flat / v_flat.norm().clamp(min=1e-12)
    Fv   = mvp_fn(v)
    rq_v = (v @ Fv).item()

    rand_rqs = []
    for _ in range(n_rand):
        r = torch.randn(d, dtype=torch.float64)
        r = r / r.norm().clamp(min=1e-12)
        Fr  = mvp_fn(r)
        rand_rqs.append((r @ Fr).item())
    rand_rqs = sorted(rand_rqs)
    pct = float(np.searchsorted(rand_rqs, rq_v, side="right")) / len(rand_rqs) * 100
    avg_rq = float(np.mean(rand_rqs))
    return pct, rq_v, avg_rq


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def hline(n=76): print("─" * n)

def reconstruction_rmse(originals, reconstructed):
    rmses = []
    for o, r in zip(originals, reconstructed):
        norm = o.norm()
        if norm > 1e-8:
            rmses.append(((o - r).norm() / norm).item())
    return float(np.mean(rmses)) if rmses else float("nan")

def variance_explained(originals, reconstructed, w_bar):
    """Fraction of inter-layer variance captured by reconstruction."""
    total_var = sum((o - w_bar).norm() ** 2 for o in originals).item()
    recon_var = sum((r - w_bar).norm() ** 2 for r in reconstructed).item()
    return recon_var / max(total_var, 1e-12)

def span_align(B1, B2):
    K = min(B1.shape[0], B2.shape[0])
    return ((B1[:K] @ B2[:K].T) ** 2).sum().item() / K


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 layer groups (same as Exp 22)
# ─────────────────────────────────────────────────────────────────────────────

def get_resnet18_groups(model):
    """
    Returns dict: {(shape): [list of conv layers with that shape]}
    Only standard 3×3 conv layers (no depthwise, no 1×1).
    """
    groups = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            shape = tuple(module.weight.shape)
            groups.setdefault(shape, []).append(module)
    return {k: v for k, v in groups.items() if len(v) >= 2}


# ─────────────────────────────────────────────────────────────────────────────
# Part A — Tucker quality sweep
# ─────────────────────────────────────────────────────────────────────────────

def part_a_quality_sweep(groups, ranks_list):
    hline()
    print("PART A — Tucker quality sweep: reconstruction RMSE vs rank r")
    hline()
    print()
    print("  Tucker factors computed from W_bar (HOSVD).")
    print("  Reconstruction: W_i ≈ Tucker^{-1}(GABE(Tucker(W_i))).")
    print()

    for shape, layers in sorted(groups.items(), key=lambda x: -np.prod(x[0])):
        D = int(np.prod(shape))
        L = len(layers)
        weights = [l.weight.detach().cpu() for l in layers]

        print(f"  Group {shape}  (L={L}, D={D})")
        print(f"  {'r':>4}  {'d_core':>8}  {'compress':>10}  "
              f"{'RMSE_Tucker':>12}  {'RMSE_TuckerGABE':>16}  {'var_expl':>10}")
        print("  " + "-" * 68)

        for r in ranks_list:
            ranks = [min(r, shape[0]), min(r, shape[1]),
                     shape[2], shape[3]]   # don't compress spatial dims
            d_core = int(np.prod(ranks))
            compress = D / d_core

            # Tucker only
            try:
                factors, cores, w_bar_core = hosvd_group(weights, ranks)
                recon_tucker = [tucker_reconstruct(c, factors) for c in cores]
                rmse_tucker  = reconstruction_rmse(weights, recon_tucker)

                # Tucker + GABE
                w_bar_g, B_flat, coeffs, _ = gabe_kernel(cores)
                K_gabe = B_flat.shape[0]
                g = GABE()
                # Reconstruct cores via GABE
                w_bar_flat = w_bar_g.reshape(1, -1).to(torch.float64)
                B_f64 = B_flat.to(torch.float64)
                c_f64 = coeffs.to(torch.float64)
                cores_recon_flat = c_f64 @ B_f64 + w_bar_flat   # (L, d_core)
                cores_recon = [cores_recon_flat[i].reshape(ranks)
                               for i in range(L)]
                recon_tg = [tucker_reconstruct(c, factors) for c in cores_recon]
                rmse_tg  = reconstruction_rmse(weights, recon_tg)

                w_bar_full = tucker_reconstruct(w_bar_core, factors)
                var_expl   = variance_explained(weights, recon_tg, w_bar_full)

                print(f"  {r:>4}  {d_core:>8}  {compress:>9.1f}×  "
                      f"{rmse_tucker:>12.5f}  {rmse_tg:>16.5f}  {var_expl:>9.1%}")
            except Exception as e:
                print(f"  {r:>4}  error: {e}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Rayleigh alignment preservation
# ─────────────────────────────────────────────────────────────────────────────

def part_b_rayleigh_preservation(groups, ranks_list, loader, device,
                                  n_grad=32, n_rand=300):
    hline()
    print("PART B — Rayleigh alignment preservation: kernel vs original space")
    hline()
    print()
    print("  For each (group, r):")
    print("  (1) Rayleigh pct of B_k in original R^D with flat Fisher")
    print("  (2) Rayleigh pct of B_k_core in kernel R^d_core with kernel Fisher")
    print("  (3) Preservation ratio = kernel_pct / original_pct")
    print(f"  n_grad={n_grad}  n_rand={n_rand}")
    print()

    results = {}

    for shape, layers in sorted(groups.items(), key=lambda x: np.prod(x[0])):
        D     = int(np.prod(shape))
        L     = len(layers)
        weights = [l.weight.detach().cpu() for l in layers]

        print(f"  ── Group {shape}  (L={L}, D={D}) ──")

        # Collect gradients in original space
        grads = collect_gradients(
            next(l.weight for l in layers).requires_grad_(False).__class__  # dummy
            if False else _get_model_for_group(layers, device),
            layers, loader, device, n_grad)

        if grads is None:
            print("  Skipping — no gradients collected.")
            continue

        # Original GABE in R^D
        g = GABE()
        w_bar_orig, B_orig, coeffs_orig, _ = g._extract_svd_components(weights)
        D_flat = int(np.prod(shape))
        B_orig_flat = B_orig.view(B_orig.shape[0], -1).to(torch.float64)
        B_orig_flat = B_orig_flat / B_orig_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        K_gabe = B_orig_flat.shape[0]

        def orig_mvp(v): return fisher_mvp_flat(v, grads)

        # Original-space percentiles
        orig_pcts = []
        for k in range(K_gabe):
            pct, rq, avg = rayleigh_percentile(B_orig_flat[k], orig_mvp, n_rand, D_flat)
            orig_pcts.append((pct, rq, avg))

        print(f"  {'r':>4}  {'d_core':>8}  "
              + "".join(f"  {'B'+str(k+1)+'_orig':>8}" for k in range(K_gabe))
              + "".join(f"  {'B'+str(k+1)+'_core':>8}" for k in range(K_gabe))
              + "".join(f"  {'pres'+str(k+1):>8}" for k in range(K_gabe)))
        print("  " + "-" * (16 + 26 * K_gabe))

        # Rank sweep
        for r in ranks_list:
            ranks = [min(r, shape[0]), min(r, shape[1]), shape[2], shape[3]]
            d_core = int(np.prod(ranks))

            try:
                factors, cores, w_bar_core = hosvd_group(weights, ranks)
                _, B_core, _, _ = gabe_kernel(cores)

                def kmvp(v): return kernel_fisher_mvp(v, grads, factors, shape)

                core_pcts = []
                for k in range(min(K_gabe, B_core.shape[0])):
                    pct, rq, avg = rayleigh_percentile(B_core[k], kmvp, n_rand, d_core)
                    core_pcts.append((pct, rq, avg))

                # Fill missing with nan
                while len(core_pcts) < K_gabe:
                    core_pcts.append((float("nan"), float("nan"), float("nan")))

                orig_str = "".join(f"  {p[0]:>7.1f}th" for p in orig_pcts)
                core_str = "".join(f"  {p[0]:>7.1f}th" for p in core_pcts)
                pres_str = "".join(
                    f"  {cp[0]/max(op[0],1e-3):>7.2f}×"
                    for op, cp in zip(orig_pcts, core_pcts))

                print(f"  {r:>4}  {d_core:>8}{orig_str}{core_str}{pres_str}")

                results[(shape, r)] = {
                    "orig_pcts": orig_pcts,
                    "core_pcts": core_pcts,
                }

            except Exception as e:
                print(f"  {r:>4}  error: {e}")

        # Blank line + minimum-r-for-90%-preservation
        print()
        print("  Minimum r for ≥90% Rayleigh preservation (B1 and B2):")
        for k in [0, 1]:
            for r in ranks_list:
                if (shape, r) in results:
                    op = results[(shape, r)]["orig_pcts"]
                    cp = results[(shape, r)]["core_pcts"]
                    if k < len(cp) and not np.isnan(cp[k][0]):
                        pres = cp[k][0] / max(op[k][0], 1e-3)
                        if pres >= 0.9:
                            print(f"    B{k+1}: r={r} sufficient "
                                  f"(preservation={pres:.2f}×)")
                            break
        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a temporary model around target layers for gradient collection
# ─────────────────────────────────────────────────────────────────────────────

_global_resnet18 = None

def _get_model_for_group(layers, device):
    """Return the ResNet-18 model (cached)."""
    global _global_resnet18
    if _global_resnet18 is None:
        _global_resnet18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        _global_resnet18.eval()
    return _global_resnet18


# ─────────────────────────────────────────────────────────────────────────────
# Part C — Cross-shape Tucker-GABE
# ─────────────────────────────────────────────────────────────────────────────

def part_c_cross_shape(groups, r_cross, loader, device, n_grad=32, n_rand=300):
    hline()
    print(f"PART C — Cross-shape Tucker-GABE at r={r_cross}")
    hline()
    print()
    print("  All ResNet-18 groups projected to the SAME kernel space")
    print(f"  Core shape: ({r_cross}, {r_cross}, kH, kW) per group.")
    print("  GABE is then applied to the combined set of cores.")
    print()

    all_cores  = []  # flat core vectors from all groups
    all_labels = []  # which group each core comes from
    all_factors = {}

    for shape, layers in sorted(groups.items(), key=lambda x: np.prod(x[0])):
        ranks   = [min(r_cross, shape[0]), min(r_cross, shape[1]),
                   shape[2], shape[3]]
        d_core  = int(np.prod(ranks))
        weights = [l.weight.detach().cpu() for l in layers]

        try:
            factors, cores, w_bar_core = hosvd_group(weights, ranks)
            all_factors[shape] = (factors, ranks)
            for i, c in enumerate(cores):
                all_cores.append(c.reshape(-1).float())
                all_labels.append(f"{shape}_{i}")
            print(f"  Group {shape}: {len(layers)} layers → {d_core}-dim cores added")
        except Exception as e:
            print(f"  Group {shape}: error ({e})")

    if len(all_cores) < 2:
        print("  Too few cores for cross-shape GABE.")
        return

    print()
    # Apply GABE to ALL cores
    g = GABE()
    w_bar, B_flat, coeffs, _ = g._extract_svd_components(all_cores)
    K = B_flat.shape[0]
    d_core = B_flat.shape[1]
    B_flat_n = B_flat / B_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)

    print(f"  Cross-shape GABE: L_total={len(all_cores)}  K={K}  d_core={d_core}")
    print()

    # Singular value spectrum
    stacked = torch.stack(all_cores)
    w_bar_f = stacked.mean(0)
    centered = stacked - w_bar_f
    U, S, Vh = torch.linalg.svd(centered.to(torch.float64), full_matrices=False)
    total_var = (S ** 2).sum().item()
    print("  Singular value spectrum (top-10 of cross-shape centered matrix):")
    print(f"  {'k':>4}  {'sigma_k':>10}  {'var_k%':>8}  {'cumvar%':>10}")
    print("  " + "-" * 38)
    cum = 0.0
    for k in range(min(10, len(S))):
        var_k = (S[k].item() ** 2) / max(total_var, 1e-12) * 100
        cum  += var_k
        print(f"  {k+1:>4}  {S[k].item():>10.4f}  {var_k:>7.1f}%  {cum:>9.1f}%")
    print()

    # Per-group reconstruction quality
    print("  Cross-shape GABE reconstruction RMSE per group:")
    print(f"  {'Group':>20}  {'L':>4}  {'RMSE':>8}  {'group_var%':>12}")
    print("  " + "-" * 50)
    idx = 0
    for shape, layers in sorted(groups.items(), key=lambda x: np.prod(x[0])):
        if shape not in all_factors:
            continue
        factors, ranks = all_factors[shape]
        weights = [l.weight.detach().cpu() for l in layers]
        L = len(layers)
        group_cores    = all_cores[idx:idx+L]
        group_coeffs   = coeffs[idx:idx+L]
        group_cores_f64 = torch.stack([c.to(torch.float64) for c in group_cores])
        w_bar_core_f64  = w_bar.to(torch.float64)
        B_f64           = B_flat_n.to(torch.float64)
        alpha_f64       = group_coeffs.to(torch.float64)
        recon_cores_flat = alpha_f64 @ B_f64 + w_bar_core_f64.unsqueeze(0)
        recon_cores = [recon_cores_flat[i].reshape(ranks)
                       for i in range(L)]
        recon_weights = [tucker_reconstruct(c.to(torch.float64), factors)
                         for c in recon_cores]
        rmse = reconstruction_rmse(weights, recon_weights)
        # Group's share of total variance
        group_var = float((S[idx:idx+L] ** 2).sum()) / max(total_var, 1e-12) * 100 \
                    if idx + L <= len(S) else float("nan")
        print(f"  {str(shape):>20}  {L:>4}  {rmse:>8.5f}  {group_var:>11.1f}%")
        idx += L
    print()

    # Rayleigh percentile in kernel space for cross-shape B_k
    print(f"  Rayleigh percentile of cross-shape B_k in kernel Fisher (n_rand={n_rand}):")
    print(f"  (Using largest group's Fisher as representative kernel Fisher)")
    print()

    # Use the largest group for Fisher
    largest_shape = max(groups.keys(), key=lambda s: np.prod(s))
    if largest_shape in all_factors:
        factors_large, ranks_large = all_factors[largest_shape]
        grads = collect_gradients(
            _get_model_for_group(groups[largest_shape], device),
            groups[largest_shape], loader, device, n_grad)

        if grads is not None:
            def kmvp_cross(v):
                return kernel_fisher_mvp(v, grads, factors_large, largest_shape)

            print(f"  {'k':>4}  {'Rayleigh_pct':>14}  {'rq':>10}  {'avg_rq':>10}")
            print("  " + "-" * 46)
            for k in range(min(K, 5)):
                pct, rq, avg = rayleigh_percentile(
                    B_flat_n[k].to(torch.float64), kmvp_cross, n_rand, d_core)
                print(f"  {k+1:>4}  {pct:>13.1f}th  {rq:>10.4f}  {avg:>10.4f}")
            print()

    # Cluster analysis: do cores from the same group cluster in GABE space?
    print("  Coefficient clustering: do same-group cores form clusters in alpha-space?")
    print()
    # Project coefficients to 2D via PCA for display
    coeff_f64 = coeffs.to(torch.float64).numpy()
    if coeff_f64.shape[1] >= 2:
        from numpy.linalg import svd as np_svd
        U_pca, S_pca, Vt_pca = np_svd(coeff_f64 - coeff_f64.mean(0), full_matrices=False)
        proj = U_pca[:, :2] * S_pca[:2]

        idx2 = 0
        print(f"  {'Group':>20}  {'PC1 range':>14}  {'PC2 range':>14}  {'centroid PC1':>14}")
        print("  " + "-" * 66)
        for shape, layers in sorted(groups.items(), key=lambda x: np.prod(x[0])):
            if shape not in all_factors:
                continue
            L = len(layers)
            p = proj[idx2:idx2+L]
            print(f"  {str(shape):>20}  "
                  f"[{p[:,0].min():+.2f}, {p[:,0].max():+.2f}]  "
                  f"[{p[:,1].min():+.2f}, {p[:,1].max():+.2f}]  "
                  f"{p[:,0].mean():>+13.2f}")
            idx2 += L
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Part D — Alignment budget: B_k energy in Tucker subspace
# ─────────────────────────────────────────────────────────────────────────────

def part_d_alignment_budget(groups, ranks_list):
    hline()
    print("PART D — Alignment budget: fraction of B_k energy in Tucker subspace")
    hline()
    print()
    print("  Tucker subspace = range(J) where J = U4⊗U3⊗U2⊗U1.")
    print("  Measures how much of each B_k lies in the Tucker subspace.")
    print("  Formula: energy_in_subspace = ||J^T B_k||² / ||B_k||²")
    print("  = ||tucker_project(B_k_reshaped, factors)||² / ||B_k||²")
    print()

    for shape, layers in sorted(groups.items(), key=lambda x: np.prod(x[0])):
        D = int(np.prod(shape))
        L = len(layers)
        weights = [l.weight.detach().cpu() for l in layers]

        # Flat-space GABE
        g = GABE()
        w_bar, B_orig, _, _ = g._extract_svd_components(weights)
        B_flat = B_orig.view(B_orig.shape[0], -1).to(torch.float64)
        B_flat = B_flat / B_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        K = B_flat.shape[0]

        print(f"  Group {shape}  (K={K})")
        print(f"  {'r':>4}  {'d_core':>8}  {'compress':>10}  "
              + "".join(f"  {'B'+str(k+1)+' energy%':>12}" for k in range(K)))
        print("  " + "-" * (24 + 14 * K))

        for r in ranks_list:
            ranks = [min(r, shape[0]), min(r, shape[1]), shape[2], shape[3]]
            d_core    = int(np.prod(ranks))
            compress  = D / d_core

            try:
                factors, _, _ = hosvd_group(weights, ranks)
                energies = []
                for k in range(K):
                    b_full = B_flat[k].reshape(shape).to(torch.float64)
                    b_proj = tucker_project(b_full, factors).reshape(-1)
                    # Rayleigh quotient of b in Tucker subspace:
                    # b^T J J^T b / b^T b = ||J^T b||^2 (since ||b||=1)
                    # J^T b = tucker_project(b) (in flat form)
                    # Then reconstruct: J (J^T b) = tucker_reconstruct(b_proj)
                    b_recon  = tucker_reconstruct(b_proj.reshape(ranks), factors).reshape(-1)
                    energy   = (B_flat[k] @ b_recon).item()  # cos sim since both unit
                    energies.append(energy)

                energy_str = "".join(f"  {e*100:>11.1f}%" for e in energies)
                print(f"  {r:>4}  {d_core:>8}  {compress:>9.1f}×{energy_str}")

            except Exception as e:
                print(f"  {r:>4}  error: {e}")

        print()
        # Min r for 90% energy
        for k in range(K):
            found = False
            for r in ranks_list:
                ranks_r = [min(r, shape[0]), min(r, shape[1]), shape[2], shape[3]]
                try:
                    factors, _, _ = hosvd_group(weights, ranks_r)
                    b_full = B_flat[k].reshape(shape)
                    b_proj = tucker_project(b_full, factors).reshape(-1)
                    b_recon = tucker_reconstruct(b_proj.reshape(ranks_r), factors).reshape(-1)
                    energy = (B_flat[k] @ b_recon).item()
                    if energy >= 0.90:
                        print(f"  B{k+1}: r={r} captures ≥90% energy "
                              f"({energy*100:.1f}%)")
                        found = True
                        break
                except Exception:
                    pass
            if not found:
                print(f"  B{k+1}: ≥90% energy not reached within tested ranks")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(ranks_list=None, r_cross=16, device="cpu",
        n_grad=32, n_rand=300, n_samples=1000, parts="ABCD"):

    if ranks_list is None:
        ranks_list = [2, 4, 8, 16, 32]

    print("=" * 76)
    print("GABE Experiment 31: Tucker-GABE — Rayleigh Alignment in Kernel Space")
    print("=" * 76)
    print(f"  ranks={ranks_list}  r_cross={r_cross}  device={device}")
    print(f"  n_grad={n_grad}  n_rand={n_rand}  parts={parts}")
    print()
    print("  QUESTION: Does Tucker projection to a shared kernel space preserve")
    print("  the Rayleigh alignment (B_k at 99th+ pct) established in Exp 8-12?")
    print()
    print("  d_core at r=16: (16, 16, 3, 3) = 2304  for ALL conv groups")
    print("  vs D: 36k / 147k / 590k / 2.4M  for ResNet-18 groups")
    print("  → Cross-shape GABE becomes tractable")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("  Loading ResNet-18 (ImageNet pretrained)...", end=" ", flush=True)
    try:
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = tvm.resnet18(pretrained=False)
        print("(random init — pretrained weights unavailable)", end=" ")
    model.eval()
    global _global_resnet18
    _global_resnet18 = model
    print("done")

    groups = get_resnet18_groups(model)
    print(f"  Found {len(groups)} 3×3 conv groups:")
    for shape, layers in sorted(groups.items(), key=lambda x: np.prod(x[0])):
        print(f"    {shape}  L={len(layers)}")
    print()

    # ── Data loader ──────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                          download=True, transform=tf)
        sub = torch.utils.data.Subset(ds, list(range(min(n_samples, len(ds)))))
        loader = torch.utils.data.DataLoader(sub, batch_size=1,
                                             shuffle=False, num_workers=0)
        print("  Data: CIFAR-10 (batch_size=1 for per-sample gradients)")
    except Exception as e:
        print(f"  CIFAR-10 unavailable ({e}). Using synthetic data.")
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(n_samples, 3, 32, 32),
                torch.randint(0, 1000, (n_samples,))),
            batch_size=1, shuffle=False, num_workers=0)
    print()

    # ── Run parts ─────────────────────────────────────────────────────────────
    if "A" in parts:
        part_a_quality_sweep(groups, ranks_list)

    if "B" in parts:
        part_b_rayleigh_preservation(groups, ranks_list, loader, device,
                                     n_grad=n_grad, n_rand=n_rand)

    if "C" in parts:
        part_c_cross_shape(groups, r_cross, loader, device,
                           n_grad=n_grad, n_rand=n_rand)

    if "D" in parts:
        part_d_alignment_budget(groups, ranks_list)

    # ── Summary ───────────────────────────────────────────────────────────────
    hline()
    print("VERDICT")
    hline()
    print()
    print("  Tucker-GABE theoretical compression:")
    for r in ranks_list:
        d_core   = r * r * 3 * 3
        compress = [int(np.prod(s)) / d_core
                    for s in groups.keys()]
        print(f"    r={r:>3}: d_core={d_core:>5}  "
              f"compress=[{min(compress):.0f}×..{max(compress):.0f}×]  "
              f"(vs D=[{min(int(np.prod(s)) for s in groups)//1000}k.."
              f"{max(int(np.prod(s)) for s in groups)//1000}k])")
    print()
    print("  KEY QUESTIONS ANSWERED BY THIS EXPERIMENT:")
    print()
    print("  Q1: Does Tucker projection preserve Rayleigh alignment?")
    print("      → See Part B: preservation ratio per r. If ≥0.90 at r=16,")
    print("        the entire Exp 8-12 evidence chain transfers to kernel space.")
    print()
    print("  Q2: At what r is the kernel sufficient?")
    print("      → See Part D: energy capture per B_k direction vs r.")
    print("        Exp 30A showed ~34% var explained at K=2 across W_K layers.")
    print("        This corresponds to an effective rank ≈ 2 in the output dim.")
    print("        Tucker at r=4..8 should capture the bulk of B_k energy.")
    print()
    print("  Q3: Does cross-shape Tucker-GABE find meaningful basis directions?")
    print("      → See Part C: Rayleigh percentile of cross-shape B_k.")
    print("        If B_k_cross ≥ 90th pct in kernel Fisher → unified basis works.")
    print()
    print("  IMPLICATIONS IF PRESERVATION ≥ 90%:")
    print("  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │  Tucker-GABE enables:                                            │")
    print("  │  1. CROSS-SHAPE GROUPING: layers with different C_out/C_in       │")
    print("  │     can share a common basis B_k in kernel space.               │")
    print("  │  2. SCALABLE FISHER: kernel Fisher MVP cost = O(N×d_core)        │")
    print("  │     vs O(N×D); at r=16, D=2.4M → d_core=2304 → 1000× speedup   │")
    print("  │  3. TRANSFER: W_bar and B_k from one resolution group can be     │")
    print("  │     projected into another group's kernel space for alignment.  │")
    print("  └──────────────────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 31: Tucker-GABE — Rayleigh Alignment in Kernel Space")
    parser.add_argument("--ranks",     type=str,  default="2,4,8,16,32",
                        help="Comma-separated Tucker ranks to sweep (default 2,4,8,16,32)")
    parser.add_argument("--r_cross",   type=int,  default=16,
                        help="Rank for cross-shape grouping in Part C (default 16)")
    parser.add_argument("--n_grad",    type=int,  default=32,
                        help="Gradient samples for Fisher MVP (default 32)")
    parser.add_argument("--n_rand",    type=int,  default=300,
                        help="Random directions for Rayleigh CDF (default 300)")
    parser.add_argument("--n_samples", type=int,  default=1000,
                        help="Dataset subset size (default 1000)")
    parser.add_argument("--device",    type=str,  default="cpu")
    parser.add_argument("--parts",     type=str,  default="ABCD",
                        help="Which parts to run: A, B, C, D or any combination")
    args = parser.parse_args()

    ranks_list = [int(r.strip()) for r in args.ranks.split(",")]
    run(ranks_list=ranks_list, r_cross=args.r_cross, device=args.device,
        n_grad=args.n_grad, n_rand=args.n_rand, n_samples=args.n_samples,
        parts=args.parts.upper())
