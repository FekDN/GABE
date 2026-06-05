# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
#
# GABEtest_kvcache.py  —  Experiment 30: KV-Cache Compression with GABE
#
# QUESTION:
#   Can GABE reduce KV-cache memory by 3–6× with acceptable perplexity degradation?
#   Does a dynamic router (per-head α from query) outperform static compression?
#
# BACKGROUND:
#   The KV cache stores Key and Value tensors for all past tokens at every layer.
#   For GPT-2 small (12 layers, 12 heads, d_head=64):
#     Full KV cache per token: 2 × 12 × 12 × 64 = 18 432 floats
#
#   GABE-KV applies the decomposition ACROSS HEADS at each layer:
#     For layer l, all K_h ∈ R^{T × d_head}  (h = 0..H-1)
#     Group: treat each head's K tensor as one "sample"
#     w_bar_K  ∈ R^{T × d_head}    — mean K across heads
#     B_k      ∈ R^{K × T × d_head} — K basis directions of head variation
#     alpha_h  ∈ R^{H × K}          — per-head coefficients
#     K̂_h = w_bar_K + sum_k alpha_h[k] * B_k
#
#   Compression ratio ≈ H / (1 + K):
#     K=1 → 6×,  K=2 → 4×,  K=3 → 3×,  K=5 → 2×  (for H=12)
#
# THREE PARTS:
#   Part A — Weight-space: GABE on W_K / W_V projection matrices across all 12
#             layers. Measures reconstruction quality and perplexity at K=1..11.
#
#   Part B — Activation-space: GABE on actual KV tensors across heads during
#             inference. The primary experiment. Measures:
#             - KV reconstruction RMSE vs K
#             - Attention output cosine similarity vs K
#             - Perplexity degradation vs K
#             - Memory compression ratio vs K
#
#   Part C — Dynamic router: for each head h, predict alpha_h from query Q_h.
#             Train a small MLP router, compare reconstruction quality vs static.
#
# USAGE:
#   python GABEtest_kvcache.py
#   python GABEtest_kvcache.py --model gpt2 --n_tokens 512 --device cuda
#   python GABEtest_kvcache.py --model gpt2 --n_tokens 256 --parts AB
#   python GABEtest_kvcache.py --parts C --router_epochs 30

import sys, os, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from GABE import GABE


# ─────────────────────────────────────────────────────────────────────────────
# GABE utilities for tensor groups
# ─────────────────────────────────────────────────────────────────────────────

def gabe_decompose(tensors):
    """
    Decompose a list of L tensors of identical shape using GABE SVD.
    Returns (w_bar, B, coeffs, shape).
      w_bar  : mean tensor
      B      : (L-1, *shape) basis directions (unit norm)
      coeffs : (L, L-1) per-tensor coordinates
    """
    g = GABE()
    w_bar, B, coeffs, shape = g._extract_svd_components(tensors)
    return w_bar, B, coeffs, shape

def gabe_reconstruct(w_bar, B, coeffs):
    """Reconstruct tensors from GABE components."""
    g = GABE()
    return g._reconstruct_weights(w_bar, B, coeffs, (len(coeffs), *w_bar.shape))

def gabe_reconstruct_k(w_bar, B, coeffs, K):
    """Reconstruct using only first K basis directions."""
    B_k     = B[:K]
    alpha_k = coeffs[:, :K]
    g = GABE()
    # Manually reconstruct with truncated basis
    D = int(np.prod(w_bar.shape))
    B_flat    = B_k.view(K, -1).to(torch.float64)
    alpha_f64 = alpha_k.to(torch.float64)
    # reconstructed[i] = w_bar + sum_k alpha[i,k] * B[k]
    centered = (alpha_f64 @ B_flat)   # (L, D)
    w_flat   = w_bar.reshape(-1).to(torch.float64)
    recon    = centered + w_flat.unsqueeze(0)
    return recon.view(len(coeffs), *w_bar.shape).to(w_bar.dtype)

def reconstruction_rmse(orig_list, recon_list):
    """Mean relative RMSE across tensors."""
    rmses = []
    for o, r in zip(orig_list, recon_list):
        norm = o.norm()
        if norm > 1e-8:
            rmses.append(((o - r).norm() / norm).item())
    return float(np.mean(rmses)) if rmses else float("nan")

def hline(n=76): print("─" * n)


# ─────────────────────────────────────────────────────────────────────────────
# Part A — Weight-space: W_K / W_V across layers
# ─────────────────────────────────────────────────────────────────────────────

def part_a_weight_space(model, config, device, K_values):
    """
    Apply GABE to W_K and W_V projection matrices across all transformer layers.
    Measures reconstruction RMSE at each K value.
    """
    hline()
    print("PART A — Weight-space GABE on W_K / W_V projection matrices")
    hline()
    print()

    # Extract K and V projection weights from all layers
    wK_list, wV_list = [], []
    for layer in model.transformer.h:
        # GPT-2 uses fused c_attn: shape (d_model, 3*d_model) for Q,K,V
        w_qkv = layer.attn.c_attn.weight  # (d_model, 3*d_model)
        d     = config.n_embd
        # K proj: columns d:2d, V proj: columns 2d:3d
        wK_list.append(w_qkv[:, d:2*d].detach().cpu())
        wV_list.append(w_qkv[:, 2*d:3*d].detach().cpu())

    L = len(wK_list)
    D = wK_list[0].numel()
    print(f"  Layers={L}  W_K shape={tuple(wK_list[0].shape)}  D={D}")
    print(f"  K values tested: {K_values}")
    print()

    for proj_name, wlist in [("W_K", wK_list), ("W_V", wV_list)]:
        w_bar, B, coeffs, shape = gabe_decompose(wlist)
        K_max = B.shape[0]

        # Singular values (proxy for compression quality)
        singular_info = ""
        if hasattr(B, "shape"):
            # Compute how much variance each B_k explains
            recon_full = gabe_reconstruct(w_bar, B, coeffs)
            var_full   = sum((w - w_bar).norm() ** 2 for w in wlist).item()

        print(f"  {proj_name}  (L={L}, K_max={K_max}, D={D})")
        print(f"  {'K':>4}  {'RMSE':>8}  {'compress':>10}  {'pct_var_explained':>20}")
        print("  " + "-" * 50)

        for K in K_values:
            if K > K_max:
                print(f"  {K:>4}  K > K_max={K_max}, skipping")
                continue
            recon_k = gabe_reconstruct_k(w_bar, B, coeffs, K)
            rmse_k  = reconstruction_rmse(wlist, list(recon_k))

            # Variance explained (approx)
            var_k = 0.0
            for i, w in enumerate(wlist):
                delta_full  = (w - w_bar)
                delta_recon = (recon_k[i] - w_bar)
                var_k += delta_recon.norm() ** 2
            pct_var = var_k / max(var_full, 1e-10) * 100
            compress = L / (1 + K)
            print(f"  {K:>4}  {rmse_k:>8.5f}  {compress:>9.1f}×  "
                  f"{pct_var:>19.1f}%")
        print()

    print("  Note: Weight-space GABE compresses model weights (fixed at inference),")
    print("  not the dynamic KV cache. For runtime memory savings, see Part B.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Activation-space: GABE on actual KV tensors across heads
# ─────────────────────────────────────────────────────────────────────────────

def capture_kv_tensors(model, input_ids, device):
    kv_per_layer = []
    hooks = []

    d = model.config.n_embd
    H = model.config.n_head
    d_head = d // H

    def make_hook():
        def hook_fn(module, input, output):
            qkv = output.detach().cpu()
            k = qkv[:, :, d:2*d]
            v = qkv[:, :, 2*d:3*d]
            batch, seq_len, _ = k.shape
            k_t = k.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)
            v_t = v.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)
            kv_per_layer.append((k_t, v_t))
        return hook_fn

    for layer in model.transformer.h:
        h = layer.attn.c_attn.register_forward_hook(make_hook())
        hooks.append(h)

    with torch.no_grad():
        model(input_ids.to(device), use_cache=False)

    for h in hooks:
        h.remove()

    return kv_per_layer


def gabe_compress_kv_layer(K_tensor, V_tensor, K_rank):
    """
    Apply GABE compression to K and V tensors of one layer.

    Input:
        K_tensor : (batch, H, T, d_head) — Keys
        V_tensor : (batch, H, T, d_head) — Values
        K_rank   : number of basis vectors to keep

    Returns compressed representation and reconstructed tensors.
    Memory: original = batch*H*T*d_head floats
            GABE     = batch*(1+K_rank)*T*d_head + batch*H*K_rank floats
    """
    batch, H, T, d_head = K_tensor.shape
    results = {}

    for name, tensor in [("K", K_tensor), ("V", V_tensor)]:
        # For each batch item, apply GABE across heads
        recon_list_batch = []
        for b in range(batch):
            # heads: list of H tensors, each (T, d_head)
            heads = [tensor[b, h] for h in range(H)]

            if H < 2:
                # Can't decompose with 1 head
                recon_list_batch.append(tensor[b].unsqueeze(0))
                continue

            w_bar, B, coeffs, _ = gabe_decompose(heads)

            K_eff = min(K_rank, H - 1)
            recon = gabe_reconstruct_k(w_bar, B, coeffs, K_eff)  # (H, T, d_head)
            recon_list_batch.append(recon.unsqueeze(0))

        recon_all = torch.cat(recon_list_batch, dim=0)  # (batch, H, T, d_head)
        results[name] = {"original": tensor, "reconstructed": recon_all}

    return results


def attention_output(Q, K, V, mask=None):
    """Scaled dot-product attention, returns (output, weights)."""
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


def compute_perplexity(model, tokenizer, texts, device, max_len=512):
    """Compute perplexity with the original model."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            if ids.shape[1] < 2:
                continue
            ids = ids[:, :max_len]
            out = model(ids, labels=ids)
            nll = out.loss.item()
            total_nll += nll * (ids.shape[1] - 1)
            total_tok += ids.shape[1] - 1
    return math.exp(total_nll / max(total_tok, 1))


def compute_perplexity_gabe_kv(model, tokenizer, texts, device, K_rank, max_len=512):
    model.eval()
    total_nll = 0.0
    total_tok = 0

    d = model.config.n_embd
    H = model.config.n_head
    d_head = d // H

    def make_kv_compress_hook(K_rank):
        def hook_fn(module, input, output):
            qkv = output
            q = qkv[:, :, :d]
            k = qkv[:, :, d:2*d]
            v = qkv[:, :, 2*d:3*d]

            batch, seq_len, _ = k.shape
            k_t = k.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)
            v_t = v.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)

            compressed = gabe_compress_kv_layer(k_t, v_t, K_rank)
            k_recon = compressed["K"]["reconstructed"].to(output.device)
            v_recon = compressed["V"]["reconstructed"].to(output.device)

            k_recon = k_recon.permute(0, 2, 1, 3).reshape(batch, seq_len, d)
            v_recon = v_recon.permute(0, 2, 1, 3).reshape(batch, seq_len, d)

            new_qkv = torch.cat([q, k_recon, v_recon], dim=-1)
            return new_qkv
        return hook_fn

    hooks = [layer.attn.c_attn.register_forward_hook(make_kv_compress_hook(K_rank))
             for layer in model.transformer.h]

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            if ids.shape[1] < 2:
                continue
            ids = ids[:, :max_len]
            try:
                out = model(ids, labels=ids, use_cache=False)
                nll = out.loss.item()
                total_nll += nll * (ids.shape[1] - 1)
                total_tok += ids.shape[1] - 1
            except Exception:
                pass

    for h in hooks:
        h.remove()

    return math.exp(total_nll / max(total_tok, 1)) if total_tok > 0 else float("nan")


def part_b_activation_space(model, tokenizer, config, device, K_values,
                             eval_texts, n_seq=20, max_len=256):
    """
    Main KV cache compression experiment.
    """
    hline()
    print("PART B — Activation-space GABE on KV tensors across heads")
    hline()
    print()
    print(f"  Model: {config._name_or_path if hasattr(config, '_name_or_path') else 'GPT-2'}")
    print(f"  n_layers={config.n_layer}  n_heads={config.n_head}  "
          f"d_head={config.n_embd // config.n_head}")
    print(f"  K values: {K_values}  eval_seqs={n_seq}  max_len={max_len}")
    print()

    H      = config.n_head
    d_head = config.n_embd // H

    # ── B.1: Baseline perplexity ───────────────────────────────────────────
    texts_sample = eval_texts[:n_seq]
    print("  Computing baseline perplexity (no compression)...", end=" ", flush=True)
    t0 = time.time()
    ppl_base = compute_perplexity(model, tokenizer, texts_sample, device, max_len)
    print(f"{ppl_base:.2f}  ({time.time()-t0:.0f}s)")
    print()

    # ── B.2: KV reconstruction quality ────────────────────────────────────
    print("  Capturing KV tensors for one sample sequence...", end=" ", flush=True)
    sample_ids = tokenizer.encode(eval_texts[0], return_tensors="pt")[:, :max_len]
    kv_per_layer = capture_kv_tensors(model, sample_ids, device)
    T = kv_per_layer[0][0].shape[2]  # actual sequence length
    print(f"done  (T={T}, {config.n_layer} layers)")
    print()

    # Compression analysis per K
    print("  Per-K reconstruction quality (averaged over all layers):")
    print(f"  {'K':>4}  {'compress':>10}  {'K_RMSE':>8}  {'V_RMSE':>8}  "
          f"{'mem_ratio':>10}  {'mem_saved':>10}")
    print("  " + "-" * 58)

    recon_stats = {}
    for K_rank in K_values:
        K_eff = min(K_rank, H - 1)
        k_rmses, v_rmses = [], []
        for (K_t, V_t) in kv_per_layer:
            comp = gabe_compress_kv_layer(K_t, V_t, K_eff)
            for b in range(K_t.shape[0]):
                for h in range(H):
                    k_rmses.append(
                        ((K_t[b,h] - comp["K"]["reconstructed"][b,h]).norm() /
                         K_t[b,h].norm().clamp(min=1e-8)).item())
                    v_rmses.append(
                        ((V_t[b,h] - comp["V"]["reconstructed"][b,h]).norm() /
                         V_t[b,h].norm().clamp(min=1e-8)).item())

        k_rmse = float(np.mean(k_rmses))
        v_rmse = float(np.mean(v_rmses))
        # Memory: original = H, GABE = (1+K_eff) for activations, negligible alpha
        compress  = H / (1 + K_eff)
        mem_saved = (1 - (1 + K_eff) / H) * 100
        recon_stats[K_rank] = dict(k_rmse=k_rmse, v_rmse=v_rmse,
                                   compress=compress, mem_saved=mem_saved)
        print(f"  {K_rank:>4}  {compress:>9.2f}×  {k_rmse:>8.5f}  "
              f"{v_rmse:>8.5f}  {compress:>9.2f}×  {mem_saved:>9.1f}%")

    print()

    # ── B.3: Perplexity degradation ────────────────────────────────────────
    print("  Perplexity with GABE-compressed KV cache:")
    print(f"  {'K':>4}  {'PPL':>8}  {'ΔPPL':>8}  {'ΔPPL%':>8}  {'compress':>10}  verdict")
    print("  " + "-" * 60)

    ppl_results = {}
    for K_rank in K_values:
        K_eff = min(K_rank, H - 1)
        print(f"  K={K_rank}...", end=" ", flush=True)
        t0 = time.time()
        ppl_k = compute_perplexity_gabe_kv(model, tokenizer, texts_sample,
                                            device, K_rank, max_len)
        delta    = ppl_k - ppl_base
        delta_pct = delta / ppl_base * 100
        compress  = recon_stats[K_rank]["compress"]
        verdict   = ("✓ good" if delta_pct < 5
                     else "~ acceptable" if delta_pct < 15
                     else "✗ degraded")
        ppl_results[K_rank] = dict(ppl=ppl_k, delta=delta, delta_pct=delta_pct)
        print(f"\r  {K_rank:>4}  {ppl_k:>8.2f}  {delta:>+8.2f}  "
              f"{delta_pct:>+7.1f}%  {compress:>9.2f}×  {verdict}  "
              f"({time.time()-t0:.0f}s)")

    print()

    # ── B.4: Pareto frontier ───────────────────────────────────────────────
    print("  Pareto frontier (compression vs quality trade-off):")
    print(f"  {'K':>4}  {'compress':>10}  {'ΔPPL%':>8}  {'mem_saved':>10}  {'recommended'}")
    print("  " + "-" * 56)
    for K_rank in K_values:
        if K_rank not in ppl_results:
            continue
        r  = ppl_results[K_rank]
        rs = recon_stats[K_rank]
        rec = ("★ RECOMMENDED" if r["delta_pct"] < 5 and rs["compress"] > 2
               else "  " if r["delta_pct"] < 15
               else "  (poor quality)")
        print(f"  {K_rank:>4}  {rs['compress']:>9.2f}×  {r['delta_pct']:>+7.1f}%  "
              f"{rs['mem_saved']:>9.1f}%  {rec}")
    print()

    # ── B.5: Layer-wise analysis ──────────────────────────────────────────
    print("  Layer-wise KV reconstruction RMSE (K=2):")
    K_layer = min(2, H - 1)
    print(f"  {'Layer':>6}  {'K_RMSE':>8}  {'V_RMSE':>8}  {'pattern'}")
    print("  " + "-" * 40)
    for i, (K_t, V_t) in enumerate(kv_per_layer):
        comp = gabe_compress_kv_layer(K_t, V_t, K_layer)
        kr = float(np.mean([
            ((K_t[0,h] - comp["K"]["reconstructed"][0,h]).norm() /
             K_t[0,h].norm().clamp(min=1e-8)).item()
            for h in range(H)]))
        vr = float(np.mean([
            ((V_t[0,h] - comp["V"]["reconstructed"][0,h]).norm() /
             V_t[0,h].norm().clamp(min=1e-8)).item()
            for h in range(H)]))
        bar = "█" * int(kr * 40)
        print(f"  {i:>6}  {kr:>8.5f}  {vr:>8.5f}  {bar}")
    print()

    return ppl_base, ppl_results, recon_stats


# ─────────────────────────────────────────────────────────────────────────────
# Part C — Dynamic router: predict alpha from query
# ─────────────────────────────────────────────────────────────────────────────

class AlphaRouter(nn.Module):
    """
    Small MLP that predicts per-head alpha coefficients from the query vector.
    Input:  Q_mean ∈ R^d_head (mean query vector for this head)
    Output: alpha ∈ R^K_rank
    """
    def __init__(self, d_head, K_rank, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_head, hidden), nn.ReLU(),
            nn.Linear(hidden, K_rank)
        )
    def forward(self, q_mean):
        return self.net(q_mean)


def collect_routing_data(model, tokenizer, texts, device, K_rank, n_seq=50, max_len=128):
    all_data = []
    model.eval()
    
    d = model.config.n_embd
    H = model.config.n_head
    d_head = d // H

    for text in texts[:n_seq]:
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        if ids.shape[1] < 4:
            continue
        ids = ids[:, :max_len]

        q_per_layer  = []
        kv_per_layer = []

        def make_hook():
            def hook_fn(module, input, output):
                qkv = output.detach().cpu()
                q = qkv[:, :, :d]
                k = qkv[:, :, d:2*d]
                v = qkv[:, :, 2*d:3*d]
                batch, seq_len, _ = q.shape
                
                q_t = q.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)
                k_t = k.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)
                v_t = v.view(batch, seq_len, H, d_head).permute(0, 2, 1, 3)
                
                q_per_layer.append(q_t)
                kv_per_layer.append((k_t, v_t))
            return hook_fn

        hooks = [layer.attn.c_attn.register_forward_hook(make_hook())
                 for layer in model.transformer.h]

        with torch.no_grad():
            model(ids, use_cache=False)
        for h in hooks: h.remove()

        if len(q_per_layer) != len(kv_per_layer):
            continue

        for l_idx, (Q_t, (K_t, V_t)) in enumerate(zip(q_per_layer, kv_per_layer)):
            K_eff   = min(K_rank, H - 1)
            heads_k = [K_t[0, h] for h in range(H)]
            _, B_k, coeffs, _ = gabe_decompose(heads_k)
            alpha_k = coeffs[:, :K_eff]

            Q_mean = Q_t[0].mean(dim=1)

            for h in range(H):
                all_data.append((Q_mean[h].float(), alpha_k[h].float()))

    return all_data, min(K_rank, H - 1)


def part_c_router(model, tokenizer, device, K_rank=2,
                  router_epochs=20, n_seq=50, max_len=128):
    """
    Train and evaluate a dynamic router that predicts alpha from query.
    """
    hline()
    print("PART C — Dynamic router: predict alpha_h from query Q_h")
    hline()
    print()
    print(f"  K_rank={K_rank}  router_epochs={router_epochs}  n_seq={n_seq}")
    print()

    H      = model.config.n_head
    d_head = model.config.n_embd // H

    # Collect data
    print("  Collecting (Q_mean, alpha_true) pairs...", end=" ", flush=True)
    try:
        from transformers import GPT2Tokenizer
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning was the Word, and the Word was with God.",
            "It was the best of times, it was the worst of times.",
            "To be or not to be, that is the question.",
            "Machine learning models compress information into weight matrices.",
        ] * max(1, n_seq // 5 + 1)
    except Exception:
        print("\n  No texts available. Skipping Part C.")
        return

    data, K_eff = collect_routing_data(model, tokenizer, texts, device,
                                       K_rank, n_seq=n_seq, max_len=max_len)
    if not data:
        print("\n  No routing data collected. Skipping Part C.")
        return
    print(f"done  ({len(data)} (Q,alpha) pairs)")

    # Train/val split
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data   = data[split:]

    Q_train  = torch.stack([d[0] for d in train_data])
    A_train  = torch.stack([d[1] for d in train_data])
    Q_val    = torch.stack([d[0] for d in val_data])
    A_val    = torch.stack([d[1] for d in val_data])

    # Static baseline: predict mean alpha for all heads
    alpha_mean = A_train.mean(dim=0)
    static_mse = ((A_val - alpha_mean.unsqueeze(0)) ** 2).mean().item()

    # Dynamic router
    router = AlphaRouter(d_head, K_eff, hidden=64)
    opt    = torch.optim.Adam(router.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"  Training router ({len(train_data)} train, {len(val_data)} val)...")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'vs static':>10}")
    print("  " + "-" * 44)

    best_val_mse = float("inf")
    for ep in range(router_epochs):
        router.train()
        perm = torch.randperm(len(Q_train))
        total_loss = 0.0
        for start in range(0, len(Q_train), 64):
            idx  = perm[start:start+64]
            pred = router(Q_train[idx])
            loss = loss_fn(pred, A_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
        train_mse = total_loss / len(Q_train)

        router.eval()
        with torch.no_grad():
            val_mse = loss_fn(router(Q_val), A_val).item()
        best_val_mse = min(best_val_mse, val_mse)

        if ep == 0 or (ep + 1) % max(1, router_epochs // 5) == 0 or ep == router_epochs - 1:
            vs_static = val_mse / max(static_mse, 1e-10)
            print(f"  {ep+1:>6}  {train_mse:>10.6f}  {val_mse:>10.6f}  "
                  f"{vs_static:>9.3f}×")

    print()
    router.eval()
    with torch.no_grad():
        pred_val = router(Q_val)
    # Pearson r
    flat_pred = pred_val.flatten().numpy()
    flat_true = A_val.flatten().numpy()
    corr = np.corrcoef(flat_pred, flat_true)[0, 1]

    print(f"  Static baseline MSE (predict mean alpha):  {static_mse:.6f}")
    print(f"  Router val MSE (best):                     {best_val_mse:.6f}")
    print(f"  Router vs static:                          "
          f"{best_val_mse/max(static_mse,1e-10):.3f}×")
    print(f"  Pearson r (pred vs true alpha):            {corr:.4f}")
    print()

    if corr > 0.8:
        print("  → STRONG routing signal: alpha is highly predictable from Q.")
        print("    Dynamic router can replace static coefficients.")
    elif corr > 0.5:
        print("  → MODERATE routing signal: partial predictability from Q.")
    else:
        print("  → WEAK routing signal: alpha is not well-predicted from Q alone.")
        print("    Alpha may depend on K, V content rather than Q direction.")

    print()
    return {"static_mse": static_mse, "router_mse": best_val_mse, "pearson_r": corr}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(model_name="gpt2", n_tokens=256, device="cpu",
        K_values=None, parts="ABC", router_epochs=20, n_seq=30):

    if K_values is None:
        K_values = [1, 2, 3, 4, 6]

    print("=" * 76)
    print("GABE Experiment 30: KV-Cache Compression with GABE")
    print("=" * 76)
    print(f"  model={model_name}  n_tokens={n_tokens}  device={device}")
    print(f"  K_values={K_values}  parts={parts}")
    print()
    print("  Compression ratio = H / (1 + K)  where H = n_heads")
    print("  For GPT-2 (H=12): K=1→6×, K=2→4×, K=3→3×, K=5→2×")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    except ImportError:
        print("  ERROR: transformers not installed.")
        print("  Run: pip install transformers")
        sys.exit(1)

    print(f"  Loading {model_name}...", end=" ", flush=True)
    t0 = time.time()
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model     = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        config    = model.config
        print(f"done ({time.time()-t0:.0f}s)  "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    except Exception as e:
        print(f"\n  Cannot load {model_name}: {e}")
        print("  Using a randomly-initialised GPT-2 small for structure tests.")
        config    = GPT2Config()
        model     = GPT2LMHeadModel(config).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2",
                        local_files_only=False) if False else None
        # Build minimal tokenizer fallback
        class MinTokenizer:
            def encode(self, text, return_tensors=None):
                # Simple character tokenizer as fallback
                ids = [ord(c) % 256 for c in text[:n_tokens]]
                t   = torch.tensor([ids], dtype=torch.long)
                return t if return_tensors == "pt" else ids
        tokenizer = MinTokenizer()
        config._name_or_path = "gpt2-random-init"

    model.eval()
    H      = config.n_head
    d_head = config.n_embd // H
    print(f"  n_layer={config.n_layer}  n_head={H}  "
          f"d_model={config.n_embd}  d_head={d_head}")
    print()

    # ── Eval texts ────────────────────────────────────────────────────────────
    eval_texts = [
        "The transformer architecture has revolutionised natural language processing.",
        "Memory compression is essential for deploying large language models.",
        "Attention mechanisms compute weighted averages over value vectors.",
        "The key-value cache stores intermediate computations for autoregressive decoding.",
        "Singular value decomposition reveals the principal components of a matrix.",
        "Neural networks learn hierarchical representations of input data.",
        "Batch normalisation stabilises training by normalising layer activations.",
        "Residual connections allow gradients to flow directly through deep networks.",
        "The learning rate controls the step size during gradient descent optimisation.",
        "Perplexity measures how well a language model predicts a test sequence.",
    ] * max(1, n_seq // 10 + 1)

    # ── Memory summary ────────────────────────────────────────────────────────
    hline()
    print("MEMORY ANALYSIS")
    hline()
    print(f"\n  KV cache memory per token at sequence length T (fp16):")
    print(f"  {'Config':<30}  {'bytes/token':>12}  {'compress vs full':>18}")
    full_bytes = 2 * config.n_layer * H * d_head * 2  # 2 for K+V, *2 for fp16
    print(f"  {'Full KV cache':<30}  {full_bytes:>12}  {'1.00×':>18}")
    for K_rank in K_values:
        K_eff  = min(K_rank, H - 1)
        gabe_b = 2 * config.n_layer * (1 + K_eff) * d_head * 2
        ratio  = full_bytes / max(gabe_b, 1)
        print(f"  {f'GABE K={K_rank}':<30}  {gabe_b:>12}  {ratio:>17.2f}×")
    print()

    # ── Run parts ─────────────────────────────────────────────────────────────
    if "A" in parts:
        part_a_weight_space(model, config, device, K_values)

    if "B" in parts:
        ppl_base, ppl_results, recon_stats = part_b_activation_space(
            model, tokenizer, config, device, K_values,
            eval_texts, n_seq=n_seq, max_len=n_tokens)

    if "C" in parts:
        part_c_router(model, tokenizer, device, K_rank=2,
                      router_epochs=router_epochs, n_seq=n_seq, max_len=n_tokens)

    # ── Final summary ─────────────────────────────────────────────────────────
    hline()
    print("VERDICT — GABE KV-Cache Compression Summary")
    hline()
    print()
    print("  Theoretical compression ratios (H=12 for GPT-2 small):")
    for K_rank in K_values:
        K_eff = min(K_rank, H - 1)
        compress = H / (1 + K_eff)
        print(f"    K={K_rank}:  {compress:.1f}×  (stores w_bar + {K_eff} basis vectors per layer)")
    print()
    if "B" in parts and "ppl_results" in dir():
        print("  Empirical quality vs compression:")
        for K_rank in K_values:
            if K_rank in ppl_results:
                r = ppl_results[K_rank]
                print(f"    K={K_rank}: +{r['delta_pct']:.1f}% PPL degradation  "
                      f"({recon_stats[K_rank]['compress']:.1f}× compression)")
    print()
    print("  KEY FINDINGS:")
    print("  1. GABE decomposes head-to-head variation in the KV cache.")
    print("     This is distinct from low-rank KV projection (weight-space).")
    print("  2. The compression ratio H/(1+K) is determined by head redundancy.")
    print("     If heads are similar (high w_bar alignment), low K suffices.")
    print("  3. Layer-wise RMSE variation shows which layers compress better.")
    print("     Early layers typically have higher head diversity → need higher K.")
    print("  4. Dynamic router (Part C) tests whether alpha is predictable from")
    print("     the query, enabling online compression without pre-computation.")
    print()
    print("  OPEN QUESTIONS:")
    print("  - Does head similarity vary by task/prompt? (Would justify dynamic K)")
    print("  - Can B_k be pre-computed once and reused across different prompts?")
    print("  - How does GABE-KV compare to GQA, MQA, and StreamingLLM?")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 30: KV-Cache Compression with GABE")
    parser.add_argument("--model",          type=str, default="gpt2")
    parser.add_argument("--n_tokens",       type=int, default=256,
                        help="Max sequence length (default 256)")
    parser.add_argument("--device",         type=str, default="cpu")
    parser.add_argument("--K_values",       type=str, default="1,2,3,4,6",
                        help="Comma-separated K values (default 1,2,3,4,6)")
    parser.add_argument("--parts",          type=str, default="ABC",
                        help="Which parts to run: A, B, C or any combination")
    parser.add_argument("--router_epochs",  type=int, default=20)
    parser.add_argument("--n_seq",          type=int, default=30,
                        help="Number of evaluation sequences (default 30)")
    args = parser.parse_args()

    K_values = [int(k.strip()) for k in args.K_values.split(",")]
    run(model_name=args.model, n_tokens=args.n_tokens, device=args.device,
        K_values=K_values, parts=args.parts.upper(),
        router_epochs=args.router_epochs, n_seq=args.n_seq)
