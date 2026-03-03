# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_llm.py — Experiment 18: LLM Validation (Critical Test)
#
# PURPOSE:
#   Validates whether GABE spectral elevation replicates on a large language model.
#   All prior experiments used CNNs (ResNet-18, Stable Diffusion).
#   If the effect is architecture-generic, it should appear in Transformer FFN/attn layers.
#
#   CRITICAL FALSIFIER: If percentile → ~50th on GPT-2, the effect is CNN-specific
#   and the "universal address space" claim does not extend to LLMs.
#
#   TESTED MODELS (in order of size):
#     - GPT-2 small  (117M)  — CPU-feasible
#     - GPT-2 medium (345M)  — CPU-feasible (slow)
#
#   LAYER GROUPS TESTED:
#     - FFN c_fc      (768 → 3072): 12 layers (one per block)
#     - FFN c_proj    (3072 → 768): 12 layers
#     - Attn c_proj   (768 → 768) : 12 layers
#
#   METRICS: Spectral percentile via empirical Fisher MVP
#   (same methodology as Experiments 9, 12)
#
# USAGE:
#   python GABEtest_llm.py
#   python GABEtest_llm.py --model gpt2-medium --n_grad 64

import sys, os
import torch
import torch.nn as nn
import numpy as np
import argparse
from collections import defaultdict
from scipy.stats import percentileofscore

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ---------------------------------------------------------------------------
# Group extraction for GPT-2
# ---------------------------------------------------------------------------

ROLE_PATTERNS = {
    "attn_c_attn":  lambda n: "c_attn" in n and "attn" in n,
    "attn_c_proj":  lambda n: "c_proj" in n and "attn" in n,
    "ffn_c_fc":     lambda n: "c_fc" in n,
    "ffn_c_proj":   lambda n: "c_proj" in n and "mlp" in n,
}


def collect_groups(model):
    """
    Returns {role: {shape: [(name, weight_tensor), ...]}}
    Uses Conv1D (transposed) or nn.Linear depending on GPT-2 variant.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for name, module in model.named_modules():
        w = None
        if hasattr(module, "weight") and module.weight is not None:
            if isinstance(module, nn.Linear):
                w = module.weight.detach()
            else:
                # Conv1D (HuggingFace GPT-2): weight is [out, in] transposed
                if module.weight.dim() == 2:
                    w = module.weight.detach().T  # normalize to [out, in]
        if w is None:
            continue
        for role, pattern in ROLE_PATTERNS.items():
            if pattern(name):
                groups[role][tuple(w.shape)].append((name, w))
                break
    return groups


def extract_basis(weights):
    gabe = GABE()
    _, B_s, _, _ = gabe._extract_svd_components(weights)
    K = B_s.shape[0];  D = B_s[0].numel()
    Q, _ = torch.linalg.qr(B_s.view(K, D).T.float())
    return Q[:, :K], D


def build_fisher_mvp(model, param, texts_encoded, n_grad, device):
    """Build Fisher MVP using text cross-entropy loss."""
    model.eval()
    grads = []
    count = 0
    for input_ids in texts_encoded:
        if count >= n_grad: break
        input_ids = input_ids.unsqueeze(0).to(device)
        model.zero_grad()
        out = model(input_ids, labels=input_ids)
        if hasattr(out, "loss") and out.loss is not None:
            out.loss.backward()
        else:
            # LM head: use log-prob of next token
            logits = out.logits[0, :-1]
            targets = input_ids[0, 1:]
            nn.CrossEntropyLoss()(logits, targets).backward()
        if param.grad is None:
            continue
        grads.append(param.grad.detach().reshape(-1).clone())
        param.grad = None
        count += 1
    if not grads:
        return None, 0.0
    G = torch.stack(grads)
    def fvp(v): return (G @ v).unsqueeze(1).mul(G).mean(0)
    return fvp, (G**2).sum(1).mean().item()


def spectral_analysis(B, fvp, D, n_samples=400, device="cpu"):
    rq_r = []
    for _ in range(n_samples):
        v = torch.randn(D, device=device); v /= v.norm()
        rq_r.append((v @ fvp(v)).item())
    rq_r = np.array(rq_r)
    K = B.shape[1]
    rq_g = np.array([(B[:, k] @ fvp(B[:, k])).item() for k in range(K)])
    pcts = np.array([percentileofscore(rq_r, r) for r in rq_g])
    ratio = rq_g.mean() / (rq_r.mean() + 1e-12)
    return rq_g, pcts, ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model_name="gpt2", n_grad=32, n_spectrum=400, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    print("=" * 62)
    print("GABE Experiment 18: LLM Validation")
    print("=" * 62)
    print(f"model={model_name}  n_grad={n_grad}  n_spectrum={n_spectrum}")
    print()

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("ERROR: pip install transformers")
        return

    print("[1] Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    # Synthetic text prompts for gradient computation
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks learn representations from data.",
        "The president announced a new policy today.",
        "Machine learning models can generalize across tasks.",
        "Language models predict the next word in a sequence.",
    ] * 20  # repeat to reach n_grad

    print("[2] Encoding text prompts...")
    encoded = []
    for p in prompts:
        ids = tokenizer.encode(p, return_tensors="pt")[0]
        if len(ids) >= 4:
            encoded.append(ids[:32])  # truncate for speed
    encoded = encoded[:n_grad + 10]

    print("[3] Collecting layer groups...")
    groups = collect_groups(model.transformer)

    print(f"\n{'Role':<20} {'Shape':<20} {'L':>4} {'K':>4} {'D':>8} "
          f"{'mean_pct':>10} {'ratio':>8}")
    print("-" * 78)

    all_results = {}
    for role in sorted(groups.keys()):
        for shape, name_weights in groups[role].items():
            if len(name_weights) < 2: continue
            weights = [w for _, w in name_weights]
            B, D = extract_basis(weights)
            K = B.shape[1]

            # Fisher MVP for the first parameter in this group
            first_name, _ = name_weights[0]
            # Find the actual parameter
            param = None
            for n, m in model.named_modules():
                if n in first_name or first_name.endswith(n):
                    if hasattr(m, "weight") and m.weight is not None:
                        param = m.weight
                        break
            if param is None:
                continue

            fvp, trace_F = build_fisher_mvp(model, param, encoded, n_grad, device)
            if fvp is None or trace_F < 1e-12:
                print(f"  {role:<20} {str(shape):<20} {len(weights):>4} {K:>4} "
                      f"{D:>8}  (no gradients)")
                continue

            rq_g, pcts, ratio = spectral_analysis(B, fvp, D, n_spectrum, device)
            label = f"{role}/{shape}"
            all_results[label] = dict(K=K, D=D, pcts=pcts, rq_g=rq_g, ratio=ratio)
            print(f"  {role:<20} {str(shape):<20} {len(weights):>4} {K:>4} "
                  f"{D:>8} {pcts.mean():>10.1f} {ratio:>8.2f}×")

    if not all_results:
        print("\nNo valid results. Check model/gradient collection.")
        return

    print()
    print("=" * 62)
    print("SUMMARY  (comparison vs small-model baseline ~79th percentile)")
    print("=" * 62)
    for label, r in sorted(all_results.items(), key=lambda x: -x[1]['pcts'].mean()):
        pct = r['pcts'].mean()
        bar = "█" * int(pct / 5)
        above99 = (r['pcts'] >= 99).sum()
        print(f"  {label:<38}  {pct:5.1f}th  {above99}/{r['K']} above 99th  {bar}")

    overall_mean = np.mean([r['pcts'].mean() for r in all_results.values()])
    print()
    print(f"Overall mean percentile: {overall_mean:.1f}th")
    print(f"(CNN baseline from Exp 12: 79th percentile)")
    print()

    if overall_mean >= 70:
        verdict = "REPLICATES — LLM percentile ≥70th. Effect generalizes to transformers."
    elif overall_mean >= 50:
        verdict = "PARTIAL — moderate elevation. Weaker than CNN baseline."
    else:
        verdict = "DOES NOT REPLICATE — effect is CNN-specific (p~50th on LLM)."
    print(f"Verdict: {verdict}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--n_grad",     type=int, default=32,
                        help="Per-sample gradients. Keep ≤64 on CPU.")
    parser.add_argument("--n_spectrum", type=int, default=400)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run(args.model, args.n_grad, args.n_spectrum, args.device, args.seed)
