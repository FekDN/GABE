# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_finetune.py — Experiment 19: Task Fine-Tuning Drift
#
# PURPOSE:
#   Tests whether the GABE basis subspace span(B) shifts when a pretrained
#   model is fine-tuned on a new task, and by how much.
#
#   Three conditions:
#     PRE:  Pretrained ResNet-18 (ImageNet weights), before any fine-tuning
#     POST: Same model after N steps of fine-tuning on CIFAR-10 subset
#     RAND: Random model (untrained baseline)
#
#   Metrics:
#     (a) Subspace alignment: span(B_pre) vs span(B_post)
#     (b) Weight drift: ||W_post - W_pre||_F per layer group
#     (c) Spectral percentile change: does functional elevation persist?
#
#   EXPECTED OUTCOMES:
#     A. span(B) drifts significantly → coefficients need re-tuning after fine-tune
#     B. span(B) is stable → fine-tuning moves only W_bar; basis is reusable
#     C. span(B) partially drifts → task-specific adjustment in higher K directions
#
#   PRACTICAL IMPLICATION for transfer learning:
#     If (B) is true, freeze B_k and only retrain α_i. This is the GABE
#     transfer learning hypothesis.
#
# USAGE:
#   python GABEtest_finetune.py
#   python GABEtest_finetune.py --ft_steps 200 --n_grad 64

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conv_group(model, target_shape):
    ws = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) == target_shape:
            ws.append(m.weight.detach().clone())
    return ws


def extract_basis(weights):
    gabe = GABE()
    _, B_s, _, _ = gabe._extract_svd_components(weights)
    K = B_s.shape[0];  D = B_s[0].numel()
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    return Q[:, :K]


def subspace_alignment(B1, B2):
    _, S, _ = torch.linalg.svd(B1.T @ B2)
    return (S ** 2).mean().item()


def build_fisher_mvp(model, param, loader, loss_fn, device, n_grad):
    model.eval()
    grads, count = [], 0
    for xb, yb in loader:
        for i in range(xb.size(0)):
            if count >= n_grad: break
            x, y = xb[i:i+1].to(device), yb[i:i+1].to(device)
            model.zero_grad(); loss_fn(model(x), y).backward()
            grads.append(param.grad.detach().reshape(-1).clone())
            param.grad = None; count += 1
        if count >= n_grad: break
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    return fvp, (G**2).sum(1).mean().item()


def spectral_percentile(B, fvp, n_samples=300, device="cpu"):
    D = B.shape[0]
    rq_r = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_r.append((v @ fvp(v)).item())
    rq_r = np.array(rq_r)
    rq_g = np.array([(B[:, k] @ fvp(B[:, k])).item() for k in range(B.shape[1])])
    return np.array([percentileofscore(rq_r, r) for r in rq_g])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    target_shape=(64, 64, 3, 3),
    ft_steps=100,
    n_grad=64,
    n_spectrum=300,
    device="cpu",
    seed=42,
):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 19: Task Fine-Tuning Drift")
    print("=" * 62)
    print(f"shape={target_shape}  ft_steps={ft_steps}  n_grad={n_grad}")
    print()

    tf224 = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf32 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf224)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, list(range(n_grad))),
        batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    # --- PRE: pretrained ---
    print("[1] Loading pretrained ResNet-18...")
    model_pre = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(device)
    model_pre.eval()
    ws_pre = get_conv_group(model_pre, target_shape)
    if len(ws_pre) < 2:
        print(f"ERROR: no group with shape {target_shape}"); return
    B_pre = extract_basis(ws_pre)
    param_pre = next(m.weight for m in model_pre.modules()
                     if isinstance(m, nn.Conv2d) and
                     tuple(m.weight.shape) == target_shape)
    fvp_pre, _ = build_fisher_mvp(model_pre, param_pre, loader, loss_fn, device, n_grad)
    pcts_pre = spectral_percentile(B_pre, fvp_pre, n_spectrum, device)
    print(f"  PRE  basis: K={B_pre.shape[1]}, D={B_pre.shape[0]}, "
          f"mean_pct={pcts_pre.mean():.1f}th")

    # --- Random baseline ---
    print("[2] Random init baseline...")
    model_rand = torchvision.models.resnet18(weights=None).to(device)
    model_rand.eval()
    ws_rand = get_conv_group(model_rand, target_shape)
    if len(ws_rand) >= 2:
        B_rand_model = extract_basis(ws_rand)
        # Use same Fisher for comparison
        pcts_rand_model = spectral_percentile(B_rand_model, fvp_pre, n_spectrum, device)
        print(f"  RAND basis: K={B_rand_model.shape[1]}, mean_pct={pcts_rand_model.mean():.1f}th")
    else:
        B_rand_model = None

    # --- Fine-tune ---
    print(f"[3] Fine-tuning {ft_steps} steps on CIFAR-10 (224px)...")
    model_post = copy.deepcopy(model_pre)
    # Replace head for 10 classes
    model_post.fc = nn.Linear(512, 10).to(device)
    opt = optim.Adam(model_post.parameters(), lr=1e-4)

    ft_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, list(range(min(ft_steps * 8, len(cifar))))),
        batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))

    model_post.train()
    step = 0
    for x, y in ft_loader:
        if step >= ft_steps: break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss_fn(model_post(x), y).backward(); opt.step()
        step += 1
    model_post.eval()

    ws_post = get_conv_group(model_post, target_shape)
    B_post = extract_basis(ws_post)
    param_post = next(m.weight for m in model_post.modules()
                      if isinstance(m, nn.Conv2d) and
                      tuple(m.weight.shape) == target_shape)
    fvp_post, _ = build_fisher_mvp(model_post, param_post, loader, loss_fn, device, n_grad)
    pcts_post = spectral_percentile(B_post, fvp_post, n_spectrum, device)
    print(f"  POST basis: K={B_post.shape[1]}, mean_pct={pcts_post.mean():.1f}th")

    # --- Metrics ---
    sa_pre_post = subspace_alignment(B_pre, B_post)
    if B_rand_model is not None:
        sa_pre_rand = subspace_alignment(B_pre, B_rand_model)
    else:
        sa_pre_rand = None

    # Weight drift per layer
    drifts = [torch.norm(ws_post[i].float() - ws_pre[i].float()).item()
              for i in range(len(ws_pre))]
    drift_mean = np.mean(drifts)
    drift_rel = drift_mean / np.mean([w.norm().item() for w in ws_pre])

    print()
    print("=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"Subspace Alignment  PRE vs POST : {sa_pre_post:.6f}")
    if sa_pre_rand:
        print(f"Subspace Alignment  PRE vs RAND : {sa_pre_rand:.6f}")
    print(f"Weight drift        ||ΔW||/||W|| : {drift_rel:.4f}  ({drift_rel*100:.1f}%)")
    print(f"Spectral percentile PRE          : {pcts_pre.mean():.1f}th")
    print(f"Spectral percentile POST         : {pcts_post.mean():.1f}th")
    pct_change = pcts_post.mean() - pcts_pre.mean()
    print(f"Percentile change                : {pct_change:+.1f}")
    print()

    rand_sa = K = B_pre.shape[1]; D = B_pre.shape[0]
    rand_sa_expected = K / D
    if sa_pre_post > 0.7:
        drift_verdict = "STABLE — span(B) preserved after fine-tuning. B_k reusable."
    elif sa_pre_post > 0.3:
        drift_verdict = "PARTIAL DRIFT — basis partially shifts. Some B_k directions change."
    else:
        drift_verdict = "LARGE DRIFT — span(B) substantially different after fine-tuning."

    print(f"Drift verdict   : {drift_verdict}")

    if abs(pct_change) < 10:
        spec_verdict = "Spectral elevation preserved after fine-tuning."
    elif pct_change < -20:
        spec_verdict = "Spectral elevation LOST after fine-tuning."
    else:
        spec_verdict = f"Spectral elevation changed by {pct_change:+.0f} points."
    print(f"Spectral verdict: {spec_verdict}")

    return dict(sa_pre_post=sa_pre_post, drift_rel=drift_rel,
                pcts_pre=pcts_pre, pcts_post=pcts_post)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",      type=int, nargs="+", default=[64, 64, 3, 3])
    parser.add_argument("--ft_steps",   type=int, default=100)
    parser.add_argument("--n_grad",     type=int, default=64)
    parser.add_argument("--n_spectrum", type=int, default=300)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run(tuple(args.shape), args.ft_steps, args.n_grad,
        args.n_spectrum, args.device, args.seed)
