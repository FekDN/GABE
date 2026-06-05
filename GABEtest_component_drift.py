#!/usr/bin/env python3
# GABEtest_component_drift.py — Experiment: GABE Component Drift Under Fine-Tuning
#
# PURPOSE:
#   Loads a pretrained DistilBERT, extracts GABE components (W_bar, B_k, alpha)
#   from the backbone weights, then compares how each component changes under
#   two fine-tuning regimes:
#
#     PRETRAINED : frozen model, baseline GABE components
#     HEAD_ONLY  : 1 epoch, only pre_classifier + classifier trainable
#                  (backbone frozen → GABE of backbone trivially unchanged;
#                   serves as sanity check that metrics report zero drift)
#     FULL_FT    : 3 epochs, all weights trainable
#                  (backbone adapts → actual measurement of per-component drift)
#
#   For each backbone group at each checkpoint, reports:
#     W_bar drift    : ||W_bar_after - W_bar_pre||_F / ||W_bar_pre||_F
#     B_k stability  : SubspaceAlign(B_pre, B_after) and ratio vs random
#     alpha drift    : per-layer ||alpha_after - alpha_pre|| in own basis
#     alpha_fixed    : alpha drift when projected onto the PRE basis
#                      (tests whether the pretrained subspace still "spans" the weights)
#     residual       : fraction of fine-tuned variation unexplained by PRE basis
#
# BACKBONE GROUPS (DistilBERT, 6 layers each):
#   q    : attention.q_lin.weight  (6 × 768 × 768)  K=5
#   k    : attention.k_lin.weight  (6 × 768 × 768)  K=5
#   v    : attention.v_lin.weight  (6 × 768 × 768)  K=5
#   out  : attention.out_lin.weight (6 × 768 × 768) K=5
#   ffn1 : ffn.lin1.weight         (6 × 3072 × 768) K=5
#   ffn2 : ffn.lin2.weight         (6 × 768 × 3072) K=5
#
# HYPOTHESIS (from GABE theory):
#   HEAD_ONLY  → backbone drift ≈ 0 for ALL metrics (sanity check)
#   FULL_FT    → B_k most stable (Exp 19: alignment 0.9996 after fine-tuning)
#              → alpha changes most (pointer hypothesis: high-leverage surface)
#              → W_bar changes moderately (shared mean shifts with task)
#   Per-group  → FFN may drift more than attention, later layers more than early

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"
K            = 3          # GABE basis rank to extract (K_eff = min(K, L-1))
N_LAYERS     = 6          # DistilBERT transformer layers
N_TRAIN      = 3000       # training subset size
MAX_SEQ_LEN  = 128
BATCH_SIZE   = 32
HEAD_EPOCHS  = 1
FT_EPOCHS    = 3
LR_HEAD      = 1e-3       # head-only LR (higher: only 2 small layers)
LR_FULL      = 2e-5       # full fine-tuning LR (standard for BERT)
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Weight group definitions ────────────────────────────────────────────────
# Template for each attention/FFN weight. {} = layer index 0..5
GROUPS = {
    "q"   : "distilbert.transformer.layer.{i}.attention.q_lin.weight",
    "k"   : "distilbert.transformer.layer.{i}.attention.k_lin.weight",
    "v"   : "distilbert.transformer.layer.{i}.attention.v_lin.weight",
    "out" : "distilbert.transformer.layer.{i}.attention.out_lin.weight",
    "ffn1": "distilbert.transformer.layer.{i}.ffn.lin1.weight",
    "ffn2": "distilbert.transformer.layer.{i}.ffn.lin2.weight",
}
GROUP_ORDER = ["q", "k", "v", "out", "ffn1", "ffn2"]


# ══════════════════════════════════════════════════════════════════════════════
# GABE core
# ══════════════════════════════════════════════════════════════════════════════

def collect_weight_stack(model, group_template):
    """Return (L, D) float32 tensor: one row per layer, flattened."""
    sd = {n: p.detach().float() for n, p in model.named_parameters()}
    rows = [sd[group_template.format(i=i)].flatten() for i in range(N_LAYERS)]
    return torch.stack(rows)          # (L, D)


def extract_gabe(W, K):
    """
    W       : (L, D)
    Returns dict with W_bar, B, alpha, S (singular values), var_explained, K_eff, D
    """
    L = W.shape[0]
    W_bar  = W.mean(0)                # (D,)
    delta  = W - W_bar                # (L, D) centered

    # Full SVD of the L×D delta matrix (L << D so cheap: O(L²D))
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)  # Vh: (L, D)

    K_eff   = min(K, L - 1, Vh.shape[0])
    B       = Vh[:K_eff].clone()      # (K, D)  top-K right singular vectors
    alpha   = delta @ B.T             # (L, K)  per-layer coordinates

    var_exp = (S[:K_eff] ** 2).sum() / (S ** 2).sum() if S.sum() > 0 else torch.tensor(0.)

    return dict(W_bar=W_bar, B=B, alpha=alpha, S=S,
                var_explained=var_exp.item(), K_eff=K_eff,
                D=W.shape[1], L=L)


def extract_all_groups(model, K):
    """Extract GABE for every backbone group. Returns dict[group_name → gabe_dict]."""
    result = {}
    for name, tmpl in GROUPS.items():
        W = collect_weight_stack(model, tmpl)
        result[name] = extract_gabe(W, K)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Comparison metrics
# ══════════════════════════════════════════════════════════════════════════════

def subspace_align(B1, B2):
    """
    SubspaceAlign(B1, B2) = (1/K) * ||B1 B2^T||_F^2
    Range [0,1], random expectation ≈ K/D.
    """
    G = B1 @ B2.T                     # (K, K)
    return (G ** 2).sum().item() / B1.shape[0]


def project_onto(W_stack, W_bar_ref, B_ref):
    """
    Given W_stack (L, D), project onto an external (reference) basis.
    Returns alpha_proj (L, K) and reconstruction residual norm (scalar).
    """
    W_bar_ref = W_bar_ref.to(W_stack.device)
    B_ref     = B_ref.to(W_stack.device)
    delta     = W_stack - W_bar_ref
    alpha_p   = delta @ B_ref.T
    recon     = W_bar_ref + alpha_p @ B_ref
    residual  = (W_stack - recon).norm() / (W_stack.norm() + 1e-10)
    return alpha_p, residual.item()


def compare_gabe(gabe_A, gabe_B):
    """
    Compare two GABE extractions for the same group.
    A = reference (e.g. PRETRAINED), B = after training.
    """
    # ── W_bar drift
    wbar_drift = ((gabe_B["W_bar"] - gabe_A["W_bar"]).norm() /
                  (gabe_A["W_bar"].norm() + 1e-10)).item()

    # ── B_k subspace alignment (1.0 = identical span)
    sa          = subspace_align(gabe_A["B"], gabe_B["B"])
    rand_base   = gabe_A["K_eff"] / gabe_A["D"]
    sa_ratio    = sa / (rand_base + 1e-20)

    # ── alpha drift in own bases (measures "pointer movement")
    # Normalise by the magnitude of alpha in A so scale doesn't dominate
    alpha_A_norm = gabe_A["alpha"].norm(dim=1).mean().item() + 1e-10
    alpha_diff   = (gabe_B["alpha"] - gabe_A["alpha"]).norm(dim=1).mean().item()
    alpha_drift  = alpha_diff / alpha_A_norm

    # ── alpha in the FIXED reference basis A
    # Reconstructs B's weights then re-projects onto A's W_bar and B
    W_B       = (gabe_B["alpha"] @ gabe_B["B"]) + gabe_B["W_bar"]  # (L, D) reconstructed
    alpha_Bin_A, residual_A = project_onto(W_B, gabe_A["W_bar"], gabe_A["B"])
    fixed_diff  = (alpha_Bin_A - gabe_A["alpha"]).norm(dim=1).mean().item()
    alpha_fixed = fixed_diff / alpha_A_norm

    # ── Per-layer alpha drift (for depth analysis)
    per_layer = [(gabe_B["alpha"][i] - gabe_A["alpha"][i]).norm().item() /
                 (gabe_A["alpha"][i].norm().item() + 1e-10)
                 for i in range(gabe_A["L"])]

    return dict(
        wbar_drift    = wbar_drift,
        subspace_align= sa,
        sa_ratio      = sa_ratio,
        rand_base     = rand_base,
        alpha_drift   = alpha_drift,
        alpha_fixed   = alpha_fixed,
        residual_A    = residual_A,
        per_layer     = per_layer,
        var_exp_A     = gabe_A["var_explained"],
        var_exp_B     = gabe_B["var_explained"],
    )


def compare_all(gabe_A, gabe_B):
    return {name: compare_gabe(gabe_A[name], gabe_B[name]) for name in gabe_A}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_sst2(tokenizer, n_train=N_TRAIN):
    raw = load_dataset("glue", "sst2")

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True,
                         padding="max_length", max_length=MAX_SEQ_LEN)

    train_ds = raw["train"].select(range(n_train)).map(tok, batched=True)
    val_ds   = raw["validation"].map(tok, batched=True)
    for ds in (train_ds, val_ds):
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_ds,   batch_size=BATCH_SIZE))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval()
    ok = total = 0
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids      = b["input_ids"].to(DEVICE),
                           attention_mask = b["attention_mask"].to(DEVICE)).logits
            ok    += (logits.argmax(-1) == b["label"].to(DEVICE)).sum().item()
            total += len(b["label"])
    return ok / total


def _train_loop(model, loader, val_loader, optimiser, scheduler, epochs, label):
    loss_fn = nn.CrossEntropyLoss()
    model.to(DEVICE).train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.
        for b in loader:
            optimiser.zero_grad()
            loss = loss_fn(
                model(input_ids      = b["input_ids"].to(DEVICE),
                      attention_mask = b["attention_mask"].to(DEVICE)).logits,
                b["label"].to(DEVICE)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if scheduler: scheduler.step()
            running += loss.item()
        val_acc = evaluate(model, val_loader)
        print(f"    [{label}  ep {ep}/{epochs}]  "
              f"loss={running/len(loader):.4f}  val_acc={val_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")
    return model


def train_head_only(model, train_loader, val_loader):
    """Freeze backbone; train pre_classifier + classifier only."""
    for name, p in model.named_parameters():
        p.requires_grad = ("classifier" in name) or ("pre_classifier" in name)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"    Trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(head only)")
    opt = torch.optim.AdamW(trainable, lr=LR_HEAD)
    return _train_loop(model, train_loader, val_loader, opt, None,
                       HEAD_EPOCHS, "HEAD_ONLY")


def train_full_ft(model, train_loader, val_loader):
    """Full fine-tuning of all parameters."""
    for p in model.parameters():
        p.requires_grad = True
    print(f"    Trainable params: {sum(p.numel() for p in model.parameters()):,}  "
          f"(all)")
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_FULL)
    total = FT_EPOCHS * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt,
                num_warmup_steps=len(train_loader), num_training_steps=total)
    return _train_loop(model, train_loader, val_loader, opt, sched,
                       FT_EPOCHS, "FULL_FT ")


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_table(metrics, title):
    print(f"\n  {'─'*78}")
    print(f"  {title}")
    print(f"  {'─'*78}")
    hdr = (f"  {'Group':5}  {'W_bar_drift':>11}  {'SA':>10}  "
           f"{'SA_ratio':>8}  {'α_drift':>8}  {'α_fixed':>8}  {'residual':>8}")
    print(hdr)
    print(f"  {'─'*78}")
    for g in GROUP_ORDER:
        m = metrics[g]
        print(f"  {g:5}  "
              f"{m['wbar_drift']:>11.6f}  "
              f"{m['subspace_align']:>10.7f}  "
              f"{m['sa_ratio']:>7.1f}×  "
              f"{m['alpha_drift']:>8.5f}  "
              f"{m['alpha_fixed']:>8.5f}  "
              f"{m['residual_A']:>8.5f}")


def print_per_layer(metrics, title):
    print(f"\n  {'─'*78}")
    print(f"  {title} — per-layer alpha drift (sorted deepest change first)")
    print(f"  {'─'*78}")
    for g in GROUP_ORDER:
        vals = metrics[g]["per_layer"]
        print(f"\n  {g}:")
        for i, v in enumerate(vals):
            bar = "█" * min(40, int(v * 60))
            print(f"    layer {i}  {v:8.5f}  {bar}")


def print_component_ranking(pre_head, pre_full):
    print(f"\n  {'═'*78}")
    print(f"  COMPONENT CHANGE RANKING")
    print(f"  (mean across groups; HEAD_ONLY should be ≈ 0, backbone frozen)")
    print(f"  {'═'*78}")

    def mu(metrics, key):
        return float(np.mean([metrics[g][key] for g in GROUP_ORDER]))

    rows = [
        ("W_bar drift",          "wbar_drift"),
        ("α drift (own basis)",  "alpha_drift"),
        ("α drift (fixed pre)",  "alpha_fixed"),
        ("B_k SA ratio",         "sa_ratio"),
        ("Residual in pre basis","residual_A"),
    ]

    print(f"\n  {'Metric':28}  {'HEAD_ONLY':>11}  {'FULL_FT':>11}  {'ratio FT/HD':>12}")
    print(f"  {'─'*68}")
    for label, key in rows:
        h = mu(pre_head, key)
        f = mu(pre_full, key)
        ratio = f / (h + 1e-10)
        suffix = "×" if key == "sa_ratio" else ""
        print(f"  {label:28}  {h:>11.5f}{suffix}  {f:>11.5f}{suffix}  {ratio:>11.1f}×")

    # Pointer hypothesis check
    print(f"\n  Pointer hypothesis (FULL_FT): "
          f"α drift / W_bar drift = "
          f"{mu(pre_full,'alpha_drift') / (mu(pre_full,'wbar_drift')+1e-10):.2f}×")

    supported = mu(pre_full, "alpha_drift") > mu(pre_full, "wbar_drift")
    print(f"  {'✓ SUPPORTED' if supported else '✗ NOT SUPPORTED'}: "
          f"α changes {'more' if supported else 'less'} than W_bar\n")

    # B_k stability verdict
    sa_full = mu(pre_full, "sa_ratio")
    stable  = sa_full > 10
    print(f"  B_k stability (FULL_FT): SA ratio = {sa_full:.1f}×  "
          f"{'✓ STABLE' if stable else '⚠ UNSTABLE'}")

    # Attention vs FFN
    attn_groups = ["q", "k", "v", "out"]
    ffn_groups  = ["ffn1", "ffn2"]
    attn_drift  = float(np.mean([pre_full[g]["alpha_drift"] for g in attn_groups]))
    ffn_drift   = float(np.mean([pre_full[g]["alpha_drift"] for g in ffn_groups]))
    print(f"\n  Attention vs FFN α drift (FULL_FT):")
    print(f"    Attention (q/k/v/out) mean α drift : {attn_drift:.5f}")
    print(f"    FFN       (ffn1/ffn2) mean α drift : {ffn_drift:.5f}")
    more = "FFN" if ffn_drift > attn_drift else "Attention"
    print(f"    → {more} adapts more ({max(attn_drift,ffn_drift)/min(attn_drift,ffn_drift):.2f}× ratio)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("GABE Experiment: Component Drift Under Fine-Tuning")
    print("=" * 80)
    print(f"  model        = {MODEL_NAME}")
    print(f"  K            = {K}  (basis rank per group)")
    print(f"  device       = {DEVICE}")
    print(f"  n_train      = {N_TRAIN}  (SST-2 subset)")
    print(f"  head_epochs  = {HEAD_EPOCHS}")
    print(f"  ft_epochs    = {FT_EPOCHS}")
    print()

    # ── Load model and tokenizer
    print("  Loading pretrained model and tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model_src  = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model_src.eval()
    n_params = sum(p.numel() for p in model_src.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M")

    # ── Load data
    print("  Loading SST-2 data...")
    train_loader, val_loader = load_sst2(tokenizer)

    # ── Evaluate pretrained accuracy
    acc_pre = evaluate(model_src.to(DEVICE), val_loader)
    print(f"  Pretrained val accuracy (before head reset): {acc_pre:.4f}")

    # Reset classifier head so HEAD_ONLY training must actually adapt
    # (makes the comparison HEAD_ONLY vs FULL_FT meaningful)
    print("\n  Resetting classifier head to random weights...")
    model_src.pre_classifier = nn.Linear(
        model_src.config.dim, model_src.config.dim)
    model_src.classifier = nn.Linear(
        model_src.config.dim, model_src.config.num_labels)
    nn.init.xavier_uniform_(model_src.pre_classifier.weight)
    nn.init.zeros_(model_src.pre_classifier.bias)
    nn.init.xavier_uniform_(model_src.classifier.weight)
    nn.init.zeros_(model_src.classifier.bias)
    acc_reset = evaluate(model_src.to(DEVICE), val_loader)
    print(f"  Val accuracy after head reset: {acc_reset:.4f}  (expected ~0.5)")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — GABE from pretrained (post-reset head, unchanged backbone)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 1 — GABE from pretrained backbone (head-reset model)")
    print("─" * 80)
    t0 = time.time()
    gabe_pre = extract_all_groups(model_src.cpu(), K)
    print(f"\n  Group   D            K_eff   var_explained")
    print(f"  {'─'*46}")
    for g in GROUP_ORDER:
        gg = gabe_pre[g]
        print(f"  {g:5}   {gg['D']:>9}    {gg['K_eff']:>3}   {gg['var_explained']:>8.2%}")
    print(f"\n  Extracted in {time.time()-t0:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — HEAD_ONLY training
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"PHASE 2 — HEAD_ONLY training ({HEAD_EPOCHS} epoch, backbone frozen)")
    print("─" * 80)
    model_head = copy.deepcopy(model_src)
    model_head = train_head_only(model_head, train_loader, val_loader)
    acc_head   = evaluate(model_head, val_loader)
    print(f"  Final HEAD_ONLY val accuracy: {acc_head:.4f}")

    model_head.cpu().eval()
    gabe_head = extract_all_groups(model_head, K)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — FULL_FT training (fresh copy of post-reset model)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"PHASE 3 — FULL_FT training ({FT_EPOCHS} epochs, all weights)")
    print("─" * 80)
    model_full = copy.deepcopy(model_src)
    model_full = train_full_ft(model_full, train_loader, val_loader)
    acc_full   = evaluate(model_full, val_loader)
    print(f"  Final FULL_FT val accuracy: {acc_full:.4f}")

    model_full.cpu().eval()
    gabe_full = extract_all_groups(model_full, K)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4 — Compare GABE components
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 4 — GABE Component Comparison")
    print("─" * 80)

    pre_vs_head = compare_all(gabe_pre, gabe_head)
    pre_vs_full = compare_all(gabe_pre, gabe_full)
    head_vs_full= compare_all(gabe_head, gabe_full)

    print_table(pre_vs_head, "PRE → HEAD_ONLY  (backbone frozen: all values should be ≈ 0)")
    print_table(pre_vs_full, "PRE → FULL_FT    (backbone fine-tuned)")
    print_table(head_vs_full,"HEAD_ONLY → FULL_FT  (additional backbone change from full FT)")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5 — Per-layer alpha drift (FULL_FT only; shows depth pattern)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 5 — Per-layer alpha drift for FULL_FT")
    print("─" * 80)
    print_per_layer(pre_vs_full, "PRE → FULL_FT")

    # ─────────────────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("VERDICT")
    print("═" * 80)
    print(f"\n  Accuracy summary:")
    print(f"    Pretrained (original head)   : {acc_pre:.4f}")
    print(f"    After head reset             : {acc_reset:.4f}")
    print(f"    HEAD_ONLY ({HEAD_EPOCHS} ep)              : {acc_head:.4f}")
    print(f"    FULL_FT   ({FT_EPOCHS} ep)              : {acc_full:.4f}")

    print_component_ranking(pre_vs_head, pre_vs_full)


if __name__ == "__main__":
    main()
