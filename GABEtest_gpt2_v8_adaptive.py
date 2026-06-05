#!/usr/bin/env python3
# GABEtest_gpt2_v8_adaptive.py — The Definitive Adaptive GABE Test

import copy, os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
N_EPOCHS    = 5
LR_FULL     = 2e-5
LR_WBAR     = 2e-5
LR_ALPHA    = 1e-4
WARMUP_FRAC = 0.1
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN     = 128
BATCH_SIZE  = 4
GEN_LEN     = 80

torch.manual_seed(SEED)
np.random.seed(SEED)

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
""".strip()

VAL_TEXT = """
The singular value decomposition provides an optimal low-rank approximation in
the Frobenius norm sense. When applied to a matrix of weight deltas from a
pretrained model the leading singular vectors correspond to principal directions
of variation across the weight group. Freezing these basis vectors during
fine-tuning constrains adaptation to remain within the subspace spanned by
pretrained variation patterns. This acts as a regularizer preventing catastrophic
forgetting while allowing the model to shift its operating point via the mean
weight and scale each direction via scalar coefficients alpha.
""".strip()

GROUPS = {
    "attn_c_proj": [f"transformer.h.{i}.attn.c_proj" for i in range(12)],
    "mlp_c_proj":  [f"transformer.h.{i}.mlp.c_proj"  for i in range(12)],
    "attn_c_attn": [f"transformer.h.{i}.attn.c_attn" for i in range(12)],
    "mlp_c_fc":    [f"transformer.h.{i}.mlp.c_fc"    for i in range(12)],
}
GROUP_ORDER = ["attn_c_proj", "mlp_c_proj", "attn_c_attn", "mlp_c_fc"]

# ══════════════════════════════════════════════════════════════════════════════
def get_weight(model, module_path):
    mod = model
    for part in module_path.split("."): mod = getattr(mod, part)
    w = mod.weight.detach().float().cpu()
    if hasattr(mod, 'nf'): w = w.T
    return w

def collect_stack(model, module_paths):
    return torch.stack([get_weight(model, path).flatten() for path in module_paths])

def extract_gabe(W):
    L = W.shape[0]; K = L - 1
    W_d = W.double()
    W_bar = W_d.mean(0)
    delta = W_d - W_bar
    _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    B = Vh[:K].clone()
    alpha = delta @ B.T
    return dict(W_bar=W_bar.float(), B=B.float(), alpha=alpha.float(), K=K, L=L, D=W.shape[1])

def extract_all(model):
    results = {}
    for g, paths in GROUPS.items():
        W = collect_stack(model, paths)
        gd = extract_gabe(W)
        mod = model
        for part in paths[0].split("."): mod = getattr(mod, part)
        gd["weight_shape"] = mod.weight.detach().shape
        gd["is_conv1d"] = hasattr(mod, 'nf')
        results[g] = gd
    return results

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
            return w_flat.view(self.weight_shape[1], self.weight_shape[0]).T   
        else:
            return w_flat.view(self.weight_shape[0], self.weight_shape[1])

class GABELinear(nn.Module):
    def __init__(self, group, layer_idx, bias_tensor=None, is_conv1d=False):
        super().__init__()
        self.group = group; self.layer_idx = layer_idx; self.is_conv1d = is_conv1d
        self.bias = nn.Parameter(bias_tensor.clone().float()) if bias_tensor is not None else None

    def forward(self, x):
        W = self.group.weight_for(self.layer_idx)
        return x @ W + (self.bias if self.bias is not None else 0) if self.is_conv1d else F.linear(x, W, self.bias)

def patch_gpt2(model, gabe_pre):
    gabe_groups = nn.ModuleDict({g: GABEGroup(gd) for g, gd in gabe_pre.items()})
    path_to_gabe = {path: (g, idx) for g, paths in GROUPS.items() for idx, path in enumerate(paths)}
    def _replace(parent_module, path_prefix):
        for child_name, child_module in list(parent_module.named_children()):
            full_path = f"{path_prefix}.{child_name}" if path_prefix else child_name
            if full_path in path_to_gabe:
                g, idx = path_to_gabe[full_path]
                bias = child_module.bias.detach() if hasattr(child_module, 'bias') and child_module.bias is not None else None
                setattr(parent_module, child_name, GABELinear(gabe_groups[g], idx, bias, hasattr(child_module, 'nf')))
            else: _replace(child_module, full_path)
    _replace(model, "")
    return model, gabe_groups

# ══════════════════════════════════════════════════════════════════════════════
def re_extract_gabe_groups(gabe_groups, old_bases):
    metrics = []
    for name, grp in gabe_groups.items():
        with torch.no_grad():
            # 1. Reconstruct current weights dynamically
            w_flat_d = grp.W_bar.detach().double() + grp.alpha.detach().double() @ grp.B.detach().double()
            
            # 2. Exact Re-SVD
            W_bar_new = w_flat_d.mean(0)
            delta = w_flat_d - W_bar_new
            _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            B_new = Vh[:grp.K].clone()
            alpha_new = delta @ B_new.T
            
            # 3. Alignment math (FIXED precedence)
            B_old = old_bases[name].double()
            align_matrix = B_old @ B_new.T
            sa = (align_matrix ** 2).sum().item() / grp.K
            cos_b1 = F.cosine_similarity(B_old[0].unsqueeze(0), B_new[0].unsqueeze(0)).item()
            
            # 4. In-place update
            grp.W_bar.data.copy_(W_bar_new.float())
            grp.alpha.data.copy_(alpha_new.float())
            grp.B.copy_(B_new.float())
            
            old_bases[name] = B_new.detach().clone()
            metrics.append((name, sa, cos_b1))
    return metrics

# ══════════════════════════════════════════════════════════════════════════════
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_len=MAX_LEN, n_samples=200):
        tokens = tokenizer.encode(text)
        self.samples = []
        step = max(1, max_len // 2)
        for start in range(0, max(1, len(tokens) - max_len), step):
            chunk = tokens[start: start + max_len]
            if len(chunk) < 8: continue
            if len(chunk) < max_len: chunk += [tokenizer.eos_token_id] * (max_len - len(chunk))
            self.samples.append(torch.tensor(chunk, dtype=torch.long))
        while len(self.samples) < n_samples: self.samples += self.samples
        self.samples = self.samples[:n_samples]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def compute_perplexity(model, loader):
    model.eval().to(DEVICE)
    total_loss = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            total_loss += model(x, labels=x).loss.item()
            n += 1
    model.cpu()
    try: return math.exp(total_loss / n)
    except: return float("inf")

def make_scheduler(opt, n_steps):
    warmup = int(n_steps * WARMUP_FRAC)
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/warmup if s < warmup else 0.5*(1.0+math.cos(math.pi*(s-warmup)/(n_steps-warmup))))

# ══════════════════════════════════════════════════════════════════════════════
def train_standard(model, train_loader, val_loader, label):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_FULL, weight_decay=1e-2)
    sched = make_scheduler(opt, len(train_loader) * N_EPOCHS)
    model.to(DEVICE)
    ppl0 = compute_perplexity(model, val_loader)
    print(f"    [{label} INIT] val_ppl={ppl0:.2f}")

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        loss_sum = 0.
        for batch in train_loader:
            x = batch.to(DEVICE)
            opt.zero_grad()
            out = model(x, labels=x)
            out.loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            sched.step()
            loss_sum += out.loss.item()
        ppl = compute_perplexity(model, val_loader)
        model.to(DEVICE)
        print(f"    [{label} ep {ep:2d}/{N_EPOCHS}]  loss={loss_sum/len(train_loader):.4f}  val_ppl={ppl:.2f}")
    model.cpu()
    return ppl

def train_gabe(model, gabe_groups, train_loader, val_loader, label, do_reextract=False):
    wbar_params  = [grp.W_bar for grp in gabe_groups.values()]
    alpha_params = [grp.alpha  for grp in gabe_groups.values()]
    bias_params  = [p for n, p in model.named_parameters() if p.requires_grad and "W_bar" not in n and "alpha" not in n]
    
    opt = torch.optim.AdamW([
        {"params": wbar_params,  "lr": LR_WBAR},
        {"params": alpha_params, "lr": LR_ALPHA},
        {"params": bias_params,  "lr": LR_ALPHA}
    ], weight_decay=1e-4)
    sched = make_scheduler(opt, len(train_loader) * N_EPOCHS)

    current_bases = {name: grp.B.clone() for name, grp in gabe_groups.items()}
    model.to(DEVICE)
    ppl0 = compute_perplexity(model, val_loader)
    print(f"    [{label} INIT] val_ppl={ppl0:.2f} (Should match BASE)")

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        loss_sum = 0.
        for batch in train_loader:
            x = batch.to(DEVICE)
            opt.zero_grad()
            out = model(x, labels=x)
            out.loss.backward()
            nn.utils.clip_grad_norm_(wbar_params + alpha_params + bias_params, 0.5)
            opt.step()
            sched.step()
            loss_sum += out.loss.item()
            
        ppl_pre = compute_perplexity(model, val_loader)
        model.to(DEVICE)
        print(f"    [{label} ep {ep:2d}/{N_EPOCHS}]  loss={loss_sum/len(train_loader):.4f}  val_ppl (pre-extract) ={ppl_pre:.2f}")
        
        if do_reextract:
            metrics = re_extract_gabe_groups(gabe_groups, current_bases)
            
            # NOTE: Re-extraction invalidates AdamW momentum (since basis rotated slightly).
            # We keep the optimizer state to see if it causes instability.
            
            ppl_post = compute_perplexity(model, val_loader)
            model.to(DEVICE)
            
            print(f"      > Re-extraction complete. val_ppl (post-extract)={ppl_post:.2f}")
            if abs(ppl_pre - ppl_post) > 0.01:
                print("      [!] WARNING: Re-extraction changed PPL! Math error!")
            
            for name, sa, cos1 in metrics:
                print(f"        {name:12}: SA={sa:.6f} | CosB1={cos1:+.5f}")

    model.cpu()
    return compute_perplexity(model, val_loader)

# ══════════════════════════════════════════════════════════════════════════════
def main():
    try: from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError: sys.exit("ERROR: pip install transformers")

    print("=" * 80)
    print("GABE v8 — Static vs Adaptive Subspace Routing Test")
    print("=" * 80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    
    train_loader = torch.utils.data.DataLoader(TextDataset(TRAIN_TEXT, tokenizer, n_samples=200), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(TextDataset(VAL_TEXT,   tokenizer, n_samples=40),  batch_size=BATCH_SIZE, shuffle=False)

    print("\n[PHASE 0] Base Model Eval...")
    ppl_base = compute_perplexity(base, val_loader)
    print(f"  BASE val_ppl: {ppl_base:.2f}")

    print("\n[PHASE 1] Extracting Base GABE (Float64)...")
    gabe_pre = extract_all(base)

    print("\n[PHASE 2] FULL_FT (Baseline)")
    m_full = copy.deepcopy(base)
    for p in m_full.parameters(): p.requires_grad_(True)
    ppl_full = train_standard(m_full, train_loader, val_loader, "FULL_FT")

    print("\n[PHASE 3] STATIC GABE (Frozen B)")
    m_static = copy.deepcopy(base)
    for p in m_static.parameters(): p.requires_grad_(False)
    m_static, gabe_groups_static = patch_gpt2(m_static, gabe_pre)
    ppl_static = train_gabe(m_static, gabe_groups_static, train_loader, val_loader, "STATIC", do_reextract=False)

    print("\n[PHASE 4] ADAPTIVE GABE (Re-extract per epoch)")
    m_adapt = copy.deepcopy(base)
    for p in m_adapt.parameters(): p.requires_grad_(False)
    m_adapt, gabe_groups_adapt = patch_gpt2(m_adapt, gabe_pre)
    ppl_adapt = train_gabe(m_adapt, gabe_groups_adapt, train_loader, val_loader, "ADAPTIVE", do_reextract=True)

    print("\n" + "=" * 80)
    print("FINAL VERDICT:")
    print(f"  BASE (No FT)     : {ppl_base:.2f}")
    print(f"  FULL_FT          : {ppl_full:.2f}")
    print(f"  STATIC GABE      : {ppl_static:.2f}")
    print(f"  ADAPTIVE GABE    : {ppl_adapt:.2f}")
    
    if ppl_adapt < ppl_static:
        print("\n  ✓ Adaptive re-extraction improved performance!")
        print("  Hypothesis: Orthogonalizing the basis aligns the loss landscape for AdamW.")
    else:
        print("\n  - Adaptive re-extraction did not improve performance.")
        print("  Hypothesis: Invalidating AdamW momentum states hurts more than orthogonalization helps.")

if __name__ == "__main__":
    main()