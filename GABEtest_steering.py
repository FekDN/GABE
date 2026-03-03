# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_steering.py — Experiment 24: Steering Vector Overlap
#
# PURPOSE:
#   Tests whether GABE basis directions B_k align with "steering vectors" —
#   directions in weight space that shift model behavior toward a target.
#
#   Steering vectors are computed as:
#     s = W_class_A - W_class_B
#   or via class-conditional mean activations projected back to weight space.
#
#   For a CNN, we use a simpler but principled construction:
#     s_c = mean_i[W_i · I(correct_class == c)] - mean_i[W_i · I(correct_class ≠ c)]
#
#   More precisely, we compute steering directions as the per-class mean
#   gradient direction:
#     s_c = (1/N_c) Σ_{x: y=c} ∂L/∂W   (class-conditional mean gradient)
#
#   METRICS:
#     (a) Cosine similarity of each steering vector with each B_k
#     (b) Projection of s_c onto span(B): ||P_B s_c|| / ||s_c||
#     (c) Comparison against projection of s_c onto random subspace of same dim
#
#   HYPOTHESIS:
#     If α_i are "pointers" that select class-relevant behavior, then steering
#     vectors (class-specific directions in weight space) should align with
#     span(B). High projection → B_k spans class-separating directions.
#
# USAGE:
#   python GABEtest_steering.py
#   python GABEtest_steering.py --shape 64 64 3 3 --n_per_class 50

import sys, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Compute class-conditional mean gradients (steering vectors)
# ---------------------------------------------------------------------------

def compute_steering_vectors(model, param, loader, loss_fn, n_classes,
                              n_per_class, device):
    """
    s_c = mean gradient w.r.t. param over samples with true label c.
    Returns (n_classes, D) tensor.
    """
    model.eval()
    D = param.numel()
    class_grads = defaultdict(list)

    for xb, yb in loader:
        for i in range(xb.size(0)):
            c = yb[i].item()
            if len(class_grads[c]) >= n_per_class:
                continue
            x = xb[i:i+1].to(device); y = yb[i:i+1].to(device)
            model.zero_grad()
            loss_fn(model(x), y).backward()
            if param.grad is not None:
                class_grads[c].append(param.grad.detach().reshape(-1).clone())
                param.grad = None

        if all(len(class_grads[c]) >= n_per_class for c in range(n_classes)):
            break

    steering = []
    for c in range(n_classes):
        if class_grads[c]:
            s_c = torch.stack(class_grads[c]).mean(0)
            s_c = s_c / (s_c.norm() + 1e-12)
            steering.append(s_c)
        else:
            steering.append(torch.zeros(D, device=device))
    return torch.stack(steering)   # (n_classes, D)


# ---------------------------------------------------------------------------
# Subspace projection
# ---------------------------------------------------------------------------

def projection_fraction(v: torch.Tensor, B: torch.Tensor) -> float:
    """||P_B v||^2 / ||v||^2  where P_B = B B^T (B is orthonormal)."""
    projections = B.T @ v      # (K,)
    return (projections ** 2).sum().item() / (v @ v).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    target_shape=(64, 64, 3, 3),
    n_per_class=50,
    n_spectrum=200,
    device="cpu",
    seed=42,
):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 24: Steering Vector Overlap")
    print("=" * 62)
    print(f"shape={target_shape}  n_per_class={n_per_class}")
    print()

    tf = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, list(range(n_per_class * 10 * 2))),
        batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    n_classes = 10

    print("[1] Loading pretrained ResNet-18...")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().to(device)
    param = next(m.weight for m in model.modules()
                 if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == target_shape)
    D = param.numel()

    print("[2] Extracting GABE basis...")
    gabe = GABE()
    ws = [m.weight.detach().to(device) for m in model.modules()
          if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == target_shape]
    _, B_s, _, _ = gabe._extract_svd_components(ws)
    K = B_s.shape[0]
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    B_gabe = Q[:, :K]
    print(f"  K={K}, D={D}")

    print(f"[3] Computing steering vectors ({n_classes} classes × {n_per_class} samples)...")
    S = compute_steering_vectors(model, param, loader, loss_fn,
                                  n_classes, n_per_class, device)  # (10, D)

    # Random baseline subspace
    A_rand = torch.randn(D, K); Q_rand, _ = torch.linalg.qr(A_rand)
    B_rand = Q_rand[:, :K]

    print()
    print(f"  {'Class':<10} {'ProjGABE':>10} {'ProjRand':>10} {'MaxCos':>10} {'BestB_k':>8}")
    print("  " + "-" * 52)

    proj_gabe_all, proj_rand_all, max_cos_all = [], [], []
    cifar_classes = ["airplane","automobile","bird","cat","deer",
                     "dog","frog","horse","ship","truck"]

    for c in range(n_classes):
        s_c = S[c]
        if s_c.norm() < 1e-8:
            print(f"  {cifar_classes[c]:<10}  (no samples)")
            continue
        pf_gabe = projection_fraction(s_c, B_gabe)
        pf_rand  = projection_fraction(s_c, B_rand)
        cosines  = (B_gabe.T @ s_c).abs()
        max_cos  = cosines.max().item()
        best_k   = cosines.argmax().item() + 1
        proj_gabe_all.append(pf_gabe);  proj_rand_all.append(pf_rand)
        max_cos_all.append(max_cos)
        print(f"  {cifar_classes[c]:<10} {pf_gabe:>10.4f} {pf_rand:>10.4f} "
              f"{max_cos:>10.4f} {best_k:>8}")

    # Bootstrap random projection baseline
    rand_projs = []
    for _ in range(n_spectrum):
        Q_r, _ = torch.linalg.qr(torch.randn(D, K)); Q_r = Q_r[:, :K]
        rand_projs.append(np.mean([projection_fraction(S[c], Q_r)
                                   for c in range(n_classes)
                                   if S[c].norm() > 1e-8]))
    rand_proj_mean = np.mean(rand_projs)

    print()
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    pg_mean = np.mean(proj_gabe_all)
    pr_mean = np.mean(proj_rand_all)
    mc_mean = np.mean(max_cos_all)
    print(f"Mean projection into span(B_GABE) : {pg_mean:.4f}")
    print(f"Mean projection into span(B_rand) : {pr_mean:.4f}")
    print(f"Bootstrap random projection mean  : {rand_proj_mean:.4f}  (K/D={K/D:.6f})")
    print(f"Ratio GABE / random               : {pg_mean/(rand_proj_mean+1e-12):.2f}×")
    print(f"Mean max cosine similarity        : {mc_mean:.4f}")
    print()

    if pg_mean > 2 * rand_proj_mean:
        verdict = ("ALIGNED — steering vectors concentrate disproportionately in span(B). "
                   "α-space is functionally class-separating.")
    elif pg_mean > 1.2 * rand_proj_mean:
        verdict = "MODERATE alignment between steering vectors and GABE basis."
    else:
        verdict = ("NO ALIGNMENT — steering vectors are orthogonal to span(B). "
                   "α-space does not capture class-conditional behavior.")
    print(f"Verdict: {verdict}")
    return dict(proj_gabe=proj_gabe_all, proj_rand=proj_rand_all,
                rand_proj_mean=rand_proj_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",       type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--n_spectrum",  type=int, default=200)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.n_per_class, args.n_spectrum,
        args.device, args.seed)
