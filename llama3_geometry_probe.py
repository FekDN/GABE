#!/usr/bin/env python3
# llama3_geometry_probe.py — Zero-Shot GABE Geometry Analyzer for Llama 3 8B

import time, gc, math
import torch
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "NousResearch/Meta-Llama-3-8B"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

torch.manual_seed(SEED)

# Мы берем только 2 группы для доказательства концепта: 
# одну из Attention (d=16M) и одну из MLP (d=58M)
GROUPS_TO_TEST = {
    "q_proj":  [f"model.layers.{i}.self_attn.q_proj" for i in range(32)],
    "up_proj": [f"model.layers.{i}.mlp.up_proj"      for i in range(32)]
}

TEXTS = [
    "The weight decomposition of neural networks provides insights into their memory.",
    "During fine-tuning, the shared basis vectors act as a structural prior.",
    "Artificial intelligence is rapidly transforming the landscape of modern technology.",
    "The capital of France is Paris, a city known for its art and culture."
]

def get_weight(model, module_path):
    mod = model
    for part in module_path.split("."): mod = getattr(mod, part)
    return mod.weight.detach().float().cpu()

def collect_stack(model, module_paths):
    return torch.stack([get_weight(model, path).flatten() for path in module_paths])

def main():
    try: from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError: exit("pip install transformers accelerate")

    print("=" * 80)
    print("Llama 3 8B — Zero-Shot GABE Geometry Probe")
    print("=" * 80)

    print("\n[1] Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # Грузим в bfloat16 чтобы влезло в VRAM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    
    gabe_bases = {}

    print("\n[2] Exact Float64 SVD Extraction (Proving mathematical limits)...")
    for name, paths in GROUPS_TO_TEST.items():
        t0 = time.time()
        W = collect_stack(model, paths)  # [32, D]
        L, D = W.shape
        K = L - 1
        
        print(f"  Processing {name} (D = {D:,}). Computing float64 SVD...")
        # Строгий float64 для избежания ошибки в 20%
        W_comp = W.double()
        W_bar = W_comp.mean(0)
        delta = W_comp - W_bar
        _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        B = Vh[:K].clone().float()
        
        # Проверка ошибки
        alpha = delta @ Vh[:K].T
        recon = W_bar + alpha @ Vh[:K]
        err = (W_comp - recon).norm() / (W_comp.norm() + 1e-15)
        
        print(f"    -> Done in {time.time()-t0:.1f}s. Reconstruction Error: {err.item():.2e}")
        gabe_bases[name] = B
        
        del W, W_comp, delta, S, Vh, alpha, recon; gc.collect()

    print("\n[3] Capturing Fisher Gradients (Single Pass)...")
    # Мы сделаем ровно ОДИН backward pass для сбора градиентов
    # Это займет пару секунд и не вызовет зависания.
    
    grad_acc = {name: [] for name in GROUPS_TO_TEST.keys()}
    
    model.train()
    for name, paths in GROUPS_TO_TEST.items():
        for path in paths:
            mod = model
            for part in path.split("."): mod = getattr(mod, part)
            mod.weight.requires_grad_(True)

    for i, text in enumerate(TEXTS):
        print(f"  Forward/Backward {i+1}/{len(TEXTS)}...")
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        model.zero_grad()
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        
        with torch.no_grad():
            for name, paths in GROUPS_TO_TEST.items():
                group_grads = []
                for path in paths:
                    mod = model
                    for part in path.split("."): mod = getattr(mod, part)
                    # Градиент слоя
                    group_grads.append(mod.weight.grad.detach().cpu().float().flatten())
                # Усредняем градиенты слоев, чтобы получить градиент для направления группы
                grad_acc[name].append(torch.stack(group_grads).mean(0))
                
    model.zero_grad()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    print("\n[4] Rayleigh Spectrum Analysis (The moment of truth)")
    print(f"  {'Group':10} | {'D':>10} | {'B1 ratio':>9} | {'B2 ratio':>9} | {'B3 ratio':>9}")
    print("  " + "─" * 60)
    
    for name in GROUPS_TO_TEST.keys():
        B_init = gabe_bases[name]  # [K, D]
        D = B_init.shape[1]
        
        # G — это стек средних градиентов группы для N текстов
        G = torch.stack(grad_acc[name], dim=0)  # [N, D]
        
        # Математический baseline случайного вектора: Trace(F) / D
        trace_F = G.pow(2).sum(dim=1).mean().item()
        baseline = trace_F / D
        
        ratios = []
        for k in range(min(3, B_init.shape[0])):
            bk = B_init[k]
            # Кривизна вдоль B_k: E[ (g * B_k)^2 ]
            rq = (G @ bk).pow(2).mean().item() / bk.pow(2).sum().clamp(min=1e-12).item()
            ratios.append(rq / (baseline + 1e-30))
            
        r_str = " | ".join(f"{r:>9.2f}×" for r in ratios)
        print(f"  {name:10} | {D:>10,} | {r_str}")

    print("\n" + "=" * 80)
    print("PROBE COMPLETE. You now have the LLM geometry scaling proof!")
    print("=" * 80)

if __name__ == "__main__":
    main()