#!/usr/bin/env python3
# GABEtest_gpt2_v5.py — GABE Fine-Tuning Test on GPT-2 (Full Report + Exact SVD)

import copy, os, sys, time, math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
N_EPOCHS    = 5
LR          = 3e-4      # для HEAD_FT
LR_FULL     = 2e-5      # для FULL_FT (понижен, чтобы избежать переобучения на 200 сэмплах)
LR_WBAR     = 2e-5      # для W̄ в GABE_FT (консервативный сдвиг центра)
LR_ALPHA    = 1e-4      # для α  в GABE_FT (навигация по базису)
WARMUP_FRAC = 0.1       # 10% шагов = warmup
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN     = 128
BATCH_SIZE  = 4
GEN_LEN     = 80

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Тексты ────────────────────────────────────────────────────────────────────
TRAIN_TEXT = """
The decomposition of weight matrices into shared basis vectors and per-layer
coefficients enables compact fine-tuning. Given a group of L weight matrices
W_1 through W_L of identical shape we compute the group mean W_bar and the
centered stack delta_W equal to W minus W_bar. Applying singular value
decomposition to delta_W yields orthonormal basis vectors B_1 through B_K
and scalar coefficients alpha such that each W_l is approximated as W_bar
plus sum over k of alpha_{lk} times B_k. During fine-tuning only W_bar and
alpha are updated while B remains frozen. This reduces trainable parameters
from L times D to D plus L times K where K equals L minus one for exact
reconstruction. The reconstruction error vanishes at K equal to L minus one
because the SVD of a centered matrix captures all variance with at most L minus
one components. Parameter savings grow with group size L and embedding dimension
D. For large transformer models with many layers of identical shape this
decomposition yields significant compression while preserving optimization
dynamics through the shared basis. The gradient flows through the formula
W equal W_bar plus alpha times B ensuring that both W_bar and alpha receive
meaningful updates from any loss. The frozen basis B acts as a structural prior
encoding directions of variation present in the pretrained weight space.
Fine-tuning corresponds to navigating this pretrained manifold via alpha while
shifting its center via W_bar. The basis vectors learned from pretrained weights
capture semantically meaningful directions such as attention patterns and factual
associations. The method scales to any layer type with matching shapes including
attention projections feed-forward layers and embedding matrices.
""".strip()

VAL_TEXT = """
The singular value decomposition provides an optimal low-rank approximation in
the Frobenius norm sense. When applied to a matrix of weight deltas from a
pretrained model the leading singular vectors correspond to principal directions
of variation across the weight group. Freezing these basis vectors during
fine-tuning constrains adaptation to remain within the subspace spanned by
pretrained variation patterns. This acts as a regularizer preventing catastrophic
forgetting while allowing the model to shift its operating point via the mean
weight and scale each direction via scalar coefficients alpha. The number of
trainable parameters scales as the embedding dimension plus the product of group
size and rank which is substantially smaller than the original count when many
layers share the same weight shape. This efficiency is particularly pronounced
in transformer architectures where attention and feed-forward layers repeat.
""".strip()

GROUPS = {
    "attn_c_proj": [f"transformer.h.{i}.attn.c_proj" for i in range(12)],
    "mlp_c_proj":  [f"transformer.h.{i}.mlp.c_proj"  for i in range(12)],
    "attn_c_attn": [f"transformer.h.{i}.attn.c_attn" for i in range(12)],
    "mlp_c_fc":    [f"transformer.h.{i}.mlp.c_fc"    for i in range(12)],
}
GROUP_ORDER = ["attn_c_proj", "mlp_c_proj", "attn_c_attn", "mlp_c_fc"]


# ══════════════════════════════════════════════════════════════════════════════
# GABE decomposition (Fixed Shapes & Exact SVD)
# ══════════════════════════════════════════════════════════════════════════════

def get_weight(model, module_path):
    mod = model
    for part in module_path.split("."):
        mod = getattr(mod, part)
    w = mod.weight.detach().float().cpu()
    # Строго по наличию атрибута 'nf' (HuggingFace Conv1D)
    if hasattr(mod, 'nf'): 
        w = w.T
    return w

def collect_stack(model, module_paths):
    weights = []
    for path in module_paths:
        w = get_weight(model, path)
        weights.append(w.flatten())
    return torch.stack(weights)

def extract_gabe(W):
    L = W.shape[0]
    K = L - 1
    # Делаем SVD в float64 для идеальной точности (убирает recon_err 1e-3)
    W_d = W.double()
    W_bar = W_d.mean(0)
    delta = W_d - W_bar
    _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    B     = Vh[:K].clone()
    alpha = delta @ B.T
    recon = W_bar + alpha @ B
    err   = (W_d - recon).norm() / (W_d.norm() + 1e-15)
    
    # Возвращаем во float32 для обучения
    return dict(W_bar=W_bar.float(), B=B.float(), alpha=alpha.float(), S=S.float(),
                recon_err=err.item(), K=K, L=L, D=W.shape[1])

def extract_all(model):
    results = {}
    for g, paths in GROUPS.items():
        W = collect_stack(model, paths)
        gd = extract_gabe(W)
        mod = model
        for part in paths[0].split("."):
            mod = getattr(mod, part)
        w0 = mod.weight.detach()
        gd["weight_shape"] = w0.shape
        gd["is_conv1d"] = hasattr(mod, 'nf')
        results[g] = gd
    return results


# ══════════════════════════════════════════════════════════════════════════════
# GABEGroup & GABELinear
# ══════════════════════════════════════════════════════════════════════════════

class GABEGroup(nn.Module):
    def __init__(self, gd):
        super().__init__()
        self.L = gd["L"]; self.K = gd["K"]; self.D = gd["D"]
        self.weight_shape = gd["weight_shape"]
        self.is_conv1d    = gd["is_conv1d"]
        self.W_bar = nn.Parameter(gd["W_bar"].clone())   
        self.alpha = nn.Parameter(gd["alpha"].clone())   
        self.register_buffer("B", gd["B"].clone())       

    def weight_for(self, layer_idx):
        w_flat = self.W_bar + self.alpha[layer_idx] @ self.B   
        if self.is_conv1d:
            # Conv1D GPT-2 хранит (in, out), поэтому мы брали .T, 
            # сейчас view как (out, in) и возвращаем .T -> (in, out)
            out_in = w_flat.view(self.weight_shape[1], self.weight_shape[0])   
            return out_in.T   
        else:
            return w_flat.view(self.weight_shape[0], self.weight_shape[1])


class GABELinear(nn.Module):
    def __init__(self, group: GABEGroup, layer_idx: int, bias_tensor=None,
                 is_conv1d=False):
        super().__init__()
        self.group     = group
        self.layer_idx = layer_idx
        self.is_conv1d = is_conv1d
        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.clone().float())
        else:
            self.bias = None

    def forward(self, x):
        W = self.group.weight_for(self.layer_idx)
        if self.is_conv1d:
            return x @ W + (self.bias if self.bias is not None else 0)
        else:
            return F.linear(x, W, self.bias)

def patch_gpt2_with_gabe(model, gabe_pre):
    gabe_groups = {}
    for g, gd in gabe_pre.items():
        gabe_groups[g] = GABEGroup(gd)

    path_to_gabe = {}
    for g, paths in GROUPS.items():
        for idx, path in enumerate(paths):
            path_to_gabe[path] = (g, idx)

    def _replace(parent_module, path_prefix):
        for child_name, child_module in list(parent_module.named_children()):
            full_path = f"{path_prefix}.{child_name}" if path_prefix else child_name
            if full_path in path_to_gabe:
                g, idx = path_to_gabe[full_path]
                grp    = gabe_groups[g]
                bias = None
                if hasattr(child_module, 'bias') and child_module.bias is not None:
                    bias = child_module.bias.detach()
                is_c1d = hasattr(child_module, 'nf') 
                new_mod = GABELinear(grp, idx, bias, is_conv1d=is_c1d)
                setattr(parent_module, child_name, new_mod)
            else:
                _replace(child_module, full_path)

    _replace(model, "")
    return model, nn.ModuleDict(gabe_groups)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_len=MAX_LEN, n_samples=200):
        tokens = tokenizer.encode(text)
        self.samples = []
        step = max(1, max_len // 2)
        for start in range(0, max(1, len(tokens) - max_len), step):
            chunk = tokens[start: start + max_len]
            if len(chunk) < 8:
                continue
            if len(chunk) < max_len:
                chunk = chunk + [tokenizer.eos_token_id] * (max_len - len(chunk))
            self.samples.append(torch.tensor(chunk, dtype=torch.long))
        while len(self.samples) < n_samples:
            self.samples += self.samples
        self.samples = self.samples[:n_samples]

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def build_loaders(tokenizer, n_train=200, n_val=40):
    train_ds = TextDataset(TRAIN_TEXT, tokenizer, n_samples=n_train)
    val_ds   = TextDataset(VAL_TEXT,   tokenizer, n_samples=n_val)
    return (torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False))

# ══════════════════════════════════════════════════════════════════════════════
# Memory
# ══════════════════════════════════════════════════════════════════════════════

def bytes_to_mb(b): return b / 1024 / 1024

def memory_mb(model):
    all_mb   = sum(p.numel() * p.element_size() for p in model.parameters())
    train_mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    buf_mb   = sum(b.numel() * b.element_size() for b in model.buffers())
    return dict(
        all_params_mb   = bytes_to_mb(all_mb),
        trainable_mb    = bytes_to_mb(train_mb),
        buffers_mb      = bytes_to_mb(buf_mb),
        total_mb        = bytes_to_mb(all_mb + buf_mb),
    )

def optimizer_memory_mb(params):
    n = sum(p.numel() for p in params if p.requires_grad)
    return bytes_to_mb(n * 4 * 2)

def activation_memory_mb(model, batch, seq_len=MAX_LEN):
    cfg = model.config if hasattr(model, 'config') else None
    if cfg is None:
        return float("nan")
    n_layers = cfg.n_layer
    hidden   = cfg.n_embd
    return bytes_to_mb(2 * n_layers * seq_len * hidden * batch * 4)

def compute_perplexity(model, loader):
    model.eval().to(DEVICE)
    total_loss = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            out = model(x, labels=x)
            total_loss += out.loss.item()
            n += 1
    model.cpu()
    avg_loss = total_loss / n
    try: return math.exp(avg_loss)
    except OverflowError: return float("inf")

# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation
# ══════════════════════════════════════════════════════════════════════════════

def make_scheduler(opt, n_steps, warmup_frac=WARMUP_FRAC):
    warmup = int(n_steps * warmup_frac)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def train(model, train_loader, val_loader, label, lr, clip=0.5, wd=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_steps = len(train_loader) * N_EPOCHS
    opt     = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched   = make_scheduler(opt, n_steps)

    mem = memory_mb(model)
    opt_mb = optimizer_memory_mb(params)
    act_mb = activation_memory_mb(model, BATCH_SIZE)

    print(f"    Обучаемых параметров : {n_train:,}")
    print(f"    Память параметров    : {mem['all_params_mb']:.1f} MB  "
          f"(обучаемые: {mem['trainable_mb']:.1f} MB, буферы: {mem['buffers_mb']:.1f} MB)")
    print(f"    Память оптимизатора  : {opt_mb:.1f} MB  (AdamW 2 момента)")
    print(f"    Активации (оценка)   : {act_mb:.1f} MB  (batch={BATCH_SIZE}, seq={MAX_LEN})")
    
    model.to(DEVICE)
    ppl0 = compute_perplexity(model, val_loader)
    print(f"    [{label} INIT] val_ppl={ppl0:.2f}  (проверка до старта)")

    hist = []
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time(); loss_sum = 0.
        for batch in train_loader:
            x = batch.to(DEVICE)
            opt.zero_grad()
            out = model(x, labels=x)
            out.loss.backward()
            nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            sched.step()
            loss_sum += out.loss.item()
        ppl = compute_perplexity(model, val_loader)
        model.to(DEVICE)
        hist.append(ppl)
        print(f"    [{label} ep {ep:2d}/{N_EPOCHS}]  "
              f"loss={loss_sum/len(train_loader):.4f}  "
              f"val_ppl={ppl:.2f}  ({time.time()-t0:.1f}s)")

    model.cpu()
    return hist, mem, opt_mb, act_mb


def train_gabe(model, gabe_groups, train_loader, val_loader, label, clip=0.5):
    wbar_params  = [grp.W_bar for grp in gabe_groups.values()]
    alpha_params = [grp.alpha  for grp in gabe_groups.values()]
    bias_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and "W_bar" not in n and "alpha" not in n]

    param_groups = [
        {"params": wbar_params,  "lr": LR_WBAR,  "name": "W_bar"},
        {"params": alpha_params, "lr": LR_ALPHA, "name": "alpha"},
    ]
    if bias_params:
        param_groups.append({"params": bias_params, "lr": LR_ALPHA, "name": "bias"})

    all_trainable = wbar_params + alpha_params + bias_params
    n_train = sum(p.numel() for p in all_trainable)
    n_steps = len(train_loader) * N_EPOCHS

    opt   = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = make_scheduler(opt, n_steps)

    mem    = memory_mb(model)
    opt_mb = optimizer_memory_mb(all_trainable)
    act_mb = activation_memory_mb(model, BATCH_SIZE)

    print(f"    Обучаемых параметров : {n_train:,}  (W̄: {sum(p.numel() for p in wbar_params):,}  "
          f"α: {sum(p.numel() for p in alpha_params):,})")
    print(f"    lm_head              : FROZEN (tied weights — не трогаем)")
    print(f"    LR W̄ / α             : {LR_WBAR} / {LR_ALPHA}")
    print(f"    Память параметров    : {mem['all_params_mb']:.1f} MB  "
          f"(обучаемые: {mem['trainable_mb']:.1f} MB, буферы: {mem['buffers_mb']:.1f} MB)")
    print(f"    Память оптимизатора  : {opt_mb:.1f} MB  (AdamW 2 момента)")

    model.to(DEVICE)
    ppl0 = compute_perplexity(model, val_loader)
    print(f"    [{label} INIT] val_ppl={ppl0:.2f}  (должно быть = BASE)")

    hist = []
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time(); loss_sum = 0.
        for batch in train_loader:
            x = batch.to(DEVICE)
            opt.zero_grad()
            out = model(x, labels=x)
            out.loss.backward()
            nn.utils.clip_grad_norm_(all_trainable, clip)
            opt.step()
            sched.step()
            loss_sum += out.loss.item()
        ppl = compute_perplexity(model, val_loader)
        model.to(DEVICE)
        hist.append(ppl)
        print(f"    [{label} ep {ep:2d}/{N_EPOCHS}]  "
              f"loss={loss_sum/len(train_loader):.4f}  "
              f"val_ppl={ppl:.2f}  ({time.time()-t0:.1f}s)")

    model.cpu()
    return hist, mem, opt_mb, act_mb

def generate_sample(model, tokenizer, prompt="The weight decomposition", n=GEN_LEN):
    model.eval().to(DEVICE)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=n,
            do_sample=True, temperature=0.85, top_p=0.92,
            pad_token_id=tokenizer.eos_token_id
        )
    model.cpu()
    return tokenizer.decode(out[0], skip_special_tokens=True)

def component_drift(gabe_pre, gabe_groups):
    rows = []
    for g in GROUP_ORDER:
        gp  = gabe_pre[g]
        grp = gabe_groups[g]
        dW  = (grp.W_bar.detach().cpu() - gp["W_bar"])
        dA  = (grp.alpha.detach().cpu() - gp["alpha"])
        rows.append(dict(
            group      = g,
            dWbar_norm = dW.norm().item(),
            dWbar_rel  = dW.norm().item() / (gp["W_bar"].norm().item() + 1e-10),
            dAlpha_rms = (dA.norm(dim=1) / (gp["alpha"].norm(dim=1) + 1e-10)).mean().item(),
        ))
    return rows

def hr(c="─"): print(c * 80)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        sys.exit("ERROR: pip install transformers")

    print("=" * 80)
    print("GABE Fine-Tuning Test v5 — GPT-2-small (Exact SVD & Fixed Shapes)")
    print("=" * 80)
    print(f"  device  = {DEVICE}")
    print(f"  epochs  = {N_EPOCHS}  |  LR_gabe = {LR_WBAR}/{LR_ALPHA}  |  LR_full = {LR_FULL}")
    print(f"  max_len = {MAX_LEN}  |  batch = {BATCH_SIZE}  |  warmup = {WARMUP_FRAC*100:.0f}%")
    print()

    print("  Загружаем GPT-2-small (pretrained)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    n_params = sum(p.numel() for p in base.parameters())
    print(f"  Параметров: {n_params/1e6:.1f}M")
    print()

    train_loader, val_loader = build_loaders(tokenizer)
    print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    memory_stats = {}

    # ── PHASE 0: Базовая генерация ────────────────────────────────────────────
    hr(); print("PHASE 0 — Базовая генерация (до fine-tuning)"); hr()
    sample_base = generate_sample(base, tokenizer)
    print(f"  Prompt: 'The weight decomposition'\n")
    print(f"  BASE GPT-2:\n  {sample_base}\n")
    ppl_base = compute_perplexity(base, val_loader)
    print(f"  Base perplexity на val: {ppl_base:.2f}")

    # ── PHASE 1: GABE-разложение ──────────────────────────────────────────────
    hr(); print("PHASE 1 — GABE Exact Decomposition (Float64)"); hr()
    gabe_pre = extract_all(base)
    param_std = param_gabe = 0
    print(f"\n  {'Group':14}  {'L':>3}  {'K':>3}  {'D':>10}  {'recon_err':>10}  "
          f"{'Std L×D':>12}  {'GABE D+LK':>12}  {'Ratio':>7}")
    print("  " + "─" * 80)
    for g in GROUP_ORDER:
        gd = gabe_pre[g]
        L, K, D = gd["L"], gd["K"], gd["D"]
        s, c = L*D, D + L*K
        param_std += s; param_gabe += c
        print(f"  {g:14}  {L:>3}  {K:>3}  {D:>10,}  {gd['recon_err']:>10.2e}  "
              f"{s:>12,}  {c:>12,}  {s/c:>6.2f}×")
    print("  " + "─" * 80)
    print(f"  {'total':14}  {'':>3}  {'':>3}  {'':>10}  {'':>10}  "
          f"{param_std:>12,}  {param_gabe:>12,}  {param_std/param_gabe:>6.2f}×")

    # ── PHASE 2: HEAD_FT ──────────────────────────────────────────────────────
    hr(); print(f"PHASE 2 — HEAD_FT  ({N_EPOCHS} эпох, только lm_head)"); hr()
    head_m = copy.deepcopy(base)
    for p in head_m.parameters(): p.requires_grad_(False)
    for p in head_m.lm_head.parameters(): p.requires_grad_(True)
    hist_head, mem_head, opt_head, act_head = train(
        head_m, train_loader, val_loader, "HEAD_FT", lr=LR)
    ppl_head = hist_head[-1]
    memory_stats["HEAD_FT"] = dict(mem=mem_head, opt=opt_head, act=act_head)
    head_m.eval()

    # ── PHASE 3: FULL_FT ──────────────────────────────────────────────────────
    hr(); print(f"PHASE 3 — FULL_FT  ({N_EPOCHS} эпох, все веса)"); hr()
    full_m = copy.deepcopy(base)
    for p in full_m.parameters(): p.requires_grad_(True)
    hist_full, mem_full, opt_full, act_full = train(
        full_m, train_loader, val_loader, "FULL_FT", lr=LR_FULL, clip=0.5, wd=1e-2)
    ppl_full = hist_full[-1]
    memory_stats["FULL_FT"] = dict(mem=mem_full, opt=opt_full, act=act_full)
    full_m.eval()

    # ── PHASE 4: GABE_FT ──────────────────────────────────────────────────────
    hr(); print(f"PHASE 4 — GABE_FT  ({N_EPOCHS} эпох, W̄ + α, B заморожен)"); hr()

    gabe_base = copy.deepcopy(base)
    for p in gabe_base.parameters(): p.requires_grad_(False)
    gabe_base, gabe_groups = patch_gpt2_with_gabe(gabe_base, gabe_pre)
    
    n_trainable = sum(p.numel() for p in gabe_base.parameters() if p.requires_grad)
    expected_trainable = param_gabe + sum(p.numel() for n, p in gabe_base.named_parameters() if 'bias' in n and p.requires_grad)
    print(f"  Проверка: обучаемых = {n_trainable:,}  "
          f"(ожидается ~{expected_trainable:,} с учетом bias)")

    hist_gabe, mem_gabe, opt_gabe, act_gabe = train_gabe(
        gabe_base, gabe_groups, train_loader, val_loader, "GABE_FT", clip=0.5)
        
    ppl_gabe = hist_gabe[-1]
    memory_stats["GABE_FT"] = dict(mem=mem_gabe, opt=opt_gabe, act=act_gabe)
    gabe_base.eval()

    # ── PHASE 5: Perplexity сравнение ─────────────────────────────────────────
    hr(); print("PHASE 5 — Perplexity на held-out тексте"); hr()
    print(f"\n  {'Модель':<20}  {'Val Perplexity':>16}  {'vs BASE':>10}  {'vs FULL':>10}")
    print("  " + "─" * 62)
    for name, ppl in [("BASE (no FT)", ppl_base),
                       ("HEAD_FT",      ppl_head),
                       ("FULL_FT",      ppl_full),
                       ("GABE_FT",      ppl_gabe)]:
        vs_base = ppl - ppl_base
        vs_full = ppl - ppl_full
        best = ppl == min(ppl_head, ppl_full, ppl_gabe)
        print(f"  {name:<20}  {ppl:>16.2f}  "
              f"{vs_base:>+10.2f}  {vs_full:>+10.2f}{'  ← лучшая' if best else ''}")

    # ── PHASE 6: Генерация образцов ───────────────────────────────────────────
    hr(); print("PHASE 6 — Генерация образцов"); hr()
    for prompt in ["The weight decomposition", "During fine-tuning"]:
        print(f"\n  Prompt: '{prompt}'")
        print(f"  FULL_FT  : {generate_sample(full_m,  tokenizer, prompt, 60)}")
        print(f"  GABE_FT  : {generate_sample(gabe_base, tokenizer, prompt, 60)}")

    # ── PHASE 7: Memory Report ────────────────────────────────────────────────
    hr("═"); print("PHASE 7 — MEMORY REPORT (честный расчёт через element_size)"); hr("═")
    print()
    print("  Примечание: tracemalloc не видит PyTorch-аллокаций (свой аллокатор).")
    print("  Все цифры считаются через p.numel() * p.element_size() (float32 = 4B).")
    print()

    col = 26
    labels = ["HEAD_FT", "FULL_FT", "GABE_FT"]
    keys_labels = [
        ("mem.all_params_mb",   "Все параметры (MB)"),
        ("mem.trainable_mb",    "Обучаемые параметры (MB)"),
        ("mem.buffers_mb",      "Буферы/frozen B (MB)"),
        ("opt",                 "Оптимизатор AdamW (MB)"),
        ("act",                 "Активации при backward (MB)"),
    ]
    def get_val(stats, key):
        if "." in key:
            a, b = key.split(".")
            return stats[a][b]
        return stats[key]

    print(f"  {'Метрика':<{col}}" + "".join(f"  {l:>12}" for l in labels))
    print("  " + "─" * (col + 3 * 14))
    for key, label in keys_labels:
        vals = [get_val(memory_stats[l], key) for l in labels]
        print(f"  {label:<{col}}" + "".join(f"  {v:>11.1f}" for v in vals))

    print("  " + "─" * (col + 3 * 14))
    totals = []
    for l in labels:
        s = memory_stats[l]
        totals.append(s["mem"]["trainable_mb"] + s["opt"] + s["act"])
    print(f"  {'ИТОГО пик (оценка, MB)':<{col}}" +
          "".join(f"  {v:>11.1f}" for v in totals))

    full_total = totals[labels.index("FULL_FT")]
    gabe_total = totals[labels.index("GABE_FT")]
    saved_mb   = full_total - gabe_total
    ratio      = full_total / max(gabe_total, 0.001)

    print()
    print(f"  Параметры GABE-групп (conv-эквивалент):")
    print(f"    Стандарт (L×D): {param_std:>12,}  параметров")
    print(f"    GABE (D+L·K):   {param_gabe:>12,}  параметров  "
          f"({param_std/param_gabe:.1f}× меньше в группах)")
    print()
    print(f"  Экономия GABE_FT vs FULL_FT:")
    print(f"    Обучаемые: {memory_stats['FULL_FT']['mem']['trainable_mb']:.1f} → "
          f"{memory_stats['GABE_FT']['mem']['trainable_mb']:.1f} MB  "
          f"(−{memory_stats['FULL_FT']['mem']['trainable_mb'] - memory_stats['GABE_FT']['mem']['trainable_mb']:.1f} MB)")
    print(f"    Оптимизатор: {memory_stats['FULL_FT']['opt']:.1f} → "
          f"{memory_stats['GABE_FT']['opt']:.1f} MB  "
          f"(−{memory_stats['FULL_FT']['opt'] - memory_stats['GABE_FT']['opt']:.1f} MB)")
    print(f"    Пик обучения: {full_total:.1f} → {gabe_total:.1f} MB  "
          f"(−{saved_mb:.1f} MB, {ratio:.1f}× экономия)")
    print()

    # ── PHASE 8: Дрейф компонент ──────────────────────────────────────────────
    hr(); print("PHASE 8 — Дрейф компонент GABE после обучения"); hr()
    drift = component_drift(gabe_pre, gabe_groups)
    print(f"\n  {'Group':14}  {'‖ΔW̄‖':>10}  {'ΔW̄/W̄₀':>10}  {'Δα/α₀ rms':>12}")
    print("  " + "─" * 52)
    for d in drift:
        print(f"  {d['group']:14}  {d['dWbar_norm']:>10.5f}  "
              f"{d['dWbar_rel']:>10.5f}  {d['dAlpha_rms']:>12.5f}")
    all_zero = all(d["dWbar_rel"] < 1e-6 and d["dAlpha_rms"] < 1e-6 for d in drift)
    if all_zero:
        print("\n  ⚠ ПРЕДУПРЕЖДЕНИЕ: все дрейфы = 0 → градиенты не текут!")
    else:
        print("\n  ✓ Компоненты обновились — граф вычислений работает корректно")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    hr("═"); print("VERDICT"); hr("═")
    gap = ppl_gabe - ppl_full
    gabe_better_head = ppl_gabe <= ppl_head

    print(f"""
  Perplexity (val, held-out text):
    BASE GPT-2 (no FT)        : {ppl_base:.2f}
    HEAD_FT   (только lm_head): {ppl_head:.2f}
    FULL_FT   (все веса)      : {ppl_full:.2f}
    GABE_FT   (W̄ + α frozen B): {ppl_gabe:.2f}   gap vs FULL = {gap:+.2f}

  Параметры GABE-групп:
    Стандарт : {param_std:,}
    GABE_FT  : {param_gabe:,}   ({param_std/param_gabe:.1f}× меньше в группах)

  Пик памяти при обучении (оценка):
    HEAD_FT  : {totals[0]:.1f} MB
    FULL_FT  : {totals[1]:.1f} MB
    GABE_FT  : {totals[2]:.1f} MB   ({ratio:.1f}× экономия vs FULL_FT)
""")

    if all_zero:
        verdict = "✗ Градиенты не текут — GABELinear не подключён к графу"
        detail  = "  Проверьте patch_gpt2_with_gabe() и forward() GABELinear."
    elif gabe_better_head and abs(gap) < 50:
        verdict = "✓ GABE_FT ≈ FULL_FT и лучше HEAD_FT — гипотеза подтверждена!"
        detail  = ("  W̄ смещает центр весового пространства к новому домену.\n"
                   "  α масштабирует каждое базисное направление по слоям.\n"
                   "  B (frozen) сохраняет геометрию предобученного пространства.")
    elif gabe_better_head:
        verdict = f"◑ GABE_FT < HEAD_FT по ppl, разрыв с FULL_FT = {gap:+.2f}"
        detail  = ("  Замороженный B ограничивает адаптацию, но работает лучше простого HEAD_FT.")
    else:
        verdict = f"✗ GABE_FT хуже HEAD_FT — gap = {ppl_gabe - ppl_head:+.2f}"
        detail  = "  Заморозка базиса оказалась слишком строгой для этого датасета."

    print(f"  {verdict}")
    print(f"\n{detail}\n")

if __name__ == "__main__":
    main()