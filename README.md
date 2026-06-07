# GABE: Groupwise Affine Basis Encoding
### A Compact Technical Overview

**Dmitry Feklin** · FeklinDN@gmail.com · 2026

> For full experiment logs, methodology, caveats, and raw numbers — see [Readme.md](./Readme.md).

---

## Abstract

**GABE** (Groupwise Affine Basis Encoding) decomposes groups of same-shaped neural network weight matrices into three components:

$$W_i = \overline{W} + \sum_{k=1}^{K} \alpha_i[k] \cdot B_k$$

| Component | Role |
|-----------|------|
| $\overline{W}$ | Shared mean weight — long-term knowledge, "RAM" |
| $B_k$ | Low-rank basis of inter-layer variation — "address space" |
| $\alpha_i$ | Per-layer coefficients — "pointers" |

The decomposition is exact at $K = L-1$ (SVD in float64, recon_err ≈ 1e-14) and compresses parameter counts 3–12× depending on group size. The basis $B_k$ is frozen during fine-tuning; only $\overline{W}$ and $\alpha$ are updated.

Experiments across ResNet-18, VGG-11, MobileNetV2, DistilBERT, and GPT-2 establish that this decomposition is not arbitrary: the leading basis directions sit above the **99th percentile** of the empirical Rayleigh spectrum in Hessian, Fisher, and Gradient Covariance matrices simultaneously, and the full method functions as a practical PEFT technique with memory savings up to **12.4×**.

---

## Key Findings

### 1. The Basis Is Not Functionally Neutral

Two of three basis directions ($B_1$, $B_2$) consistently exceed the **99th percentile** of the empirical Rayleigh spectrum across three geometrically independent matrices — Hessian ($H$), Fisher Information ($F$), and Gradient Covariance (GCM):

| Direction | Percentile (CNNs) | λ/avg_eig (H / F / GCM) |
|-----------|:-----------------:|:------------------------:|
| $B_1$ | **100th** | 10.8× / 3.0× / 2.6× |
| $B_2$ | **100th** | 7.6× / 4.6× / 4.8× |
| $B_3$ | **~35th** | ≈ random |

**Exp 32 (Llama 3 8B) confirms architecture generality** — the same bimodal structure holds on transformers at D up to 58.7M:

| Group | D | B1 ratio | B2 ratio | B3 ratio |
|-------|---|:--------:|:--------:|:--------:|
| `q_proj` | 16,777,216 | **6.50×** | **2.02×** | 1.43× |
| `up_proj` | 58,720,256 | **3.97×** | **5.43×** | 1.09× |

The effect scales from D=36k (ResNet-18 l1) to D=58M (Llama 3 8B up_proj) without attenuation. B₃ at 1.09× is statistically indistinguishable from random noise — confirming effective functional rank ≈ 2 is universal across CNNs and LLMs.

Mean spectral position across CNN experiments: **79th percentile**, spread < 2% across all three matrices. The effect scales with dimension D (3.64× at D=2304 → 26.69× at D=147456), is absent at random initialization (57.8th percentile), and emerges within the first training epoch.

The SVD rank order predicts the functional rank order. The "2–3× random" headline from aggregate experiments was a conservative average; the true picture is bimodal.

### 2. The Basis Is Stable — Across Fine-Tuning, Seeds, and Architectures

- **Fine-tuning stability:** `span(B)` subspace alignment = **0.9996** after 100 fine-tuning steps on a new task (Exp 19). The basis does not need to be retrained when adapting to a new domain.
- **Cross-architecture universality (Exp 26b):** When two models are trained from the same random seed (same initialization + same data order), span(B) alignment is **2387× above random** regardless of architecture (Plain/BN/Skip). When seeds differ, alignment returns to the random baseline (1×). Universality reflects a shared optimisation trajectory, not an architectural invariant.
- **Cross-architecture spectral consistency (Exp 22):** Spectral elevation holds across ResNet-18, VGG-11, and MobileNetV2. Mean percentile: 92.8th / 100.0th / 90.1th. Depthwise convolutions (no cross-channel mixing) are the exception at ~61st percentile.

### 3. α Are High-Leverage Pointers

Perturbing $\alpha$ with relative noise breaks model predictions at **4× lower noise** than perturbing $\overline{W}$ (ε₅₀ ratio 4×, KL ratio 18×, Exp 20b). Zero-ing, scaling, swapping, or shuffling $\alpha$ causes immediate output collapse while equivalent perturbations to $\overline{W}$ cause only gradual degradation.

This "broken pointer" behavior is geometrically grounded: $\alpha_i$ encodes projections onto $B_1$ and $B_2$, which are 100th-percentile curvature directions. Perturbing $\alpha$ displaces the model along the most functionally sensitive directions available.

### 4. W̄ Carries the Adaptation Signal During Fine-Tuning

Across three independent fine-tuning experiments (DistilBERT/SST-2, ResNet-18 4-shot, GPT-2 domain adaptation):

- $\overline{W}$ drift is the dominant signal (49M× larger than head-only baseline in DistilBERT)
- $\alpha$ drift is negligible (~0.001 relative), despite $\alpha$ being the "pointer" in inference
- $B_k$ remains structurally stable (SA ratio 394,261× above random after full fine-tuning)

This suggests a future **WBAR_ONLY** training mode — freezing both $B_k$ and $\alpha$, updating only $\overline{W}$ — could match GABE_FT performance with even greater compression.

### 5. GABE as PEFT — Validated on CNN and LLM

**ResNet-18, 4-shot new-class learning (FT-CV):**

| Method | Params | KL vs FULL_FT |
|--------|:------:|:-------------:|
| HEAD_FT | 1,026 | 0.0150 |
| FULL_FT | 11.2M | — |
| **GABE_FT** | **3.1M (3×↓)** | **0.000002** |

GABE_FT reproduces FULL_FT logits almost exactly (KL ≈ 0) with 3× fewer parameters.

**GPT-2-small, 200-sample domain adaptation (FT-LM):**

| Method | Trainable Params | Val PPL | Peak VRAM |
|--------|:----------------:|:-------:|:---------:|
| BASE (no FT) | 0 | 148.43 | — |
| HEAD_FT | 38.5M | 838.58 | 478 MB |
| FULL_FT | 124.4M | 646.40 | 1,460 MB |
| **GABE_FT** | **7.1M** | **116.47** | **118 MB** |

On small datasets FULL_FT overfits catastrophically (PPL 646). GABE_FT — by freezing $B_k$ — acts as a structural regularizer, achieving **best PPL** at **12.4× lower memory**.

### 6. Lossless Format Conversion — Zero Runtime Footprint

Re-extracting the GABE basis after training (FT-LM2) produces subspace alignment SA = **1.000000** and cosine similarity CosB₁ = **+1.000** in all groups, with zero perplexity change (identical to hundredths decimal).

> **A model trained in GABE weight-space can be saved as a standard `.safetensors` file with zero quality penalty.** No GABE-aware runtime is required for inference. GABE is a pure training-time optimization.

### 7. The B₃ Phenomenon — Effective Functional Rank ≈ 2

Across all architectures and widths tested, whenever $K \geq 3$:
- $B_1$, $B_2$ → 99–100th percentile Rayleigh quotient
- $B_3$ → ~35th percentile (statistically indistinguishable from random)

For practical PEFT and compression, **truncating to K = 2** is recommended. Including $B_3$ adds parameters and compute without capturing meaningful functional curvature.

### 8. KV-Cache Routing Signal Confirmed, Compression Infeasible

Activation-space GABE on GPT-2's KV cache degrades perplexity by **+628% at minimum** (K=6, 1.7× compression) — not viable. However, predicting $\alpha$ from the query $Q$ via a small router achieves **Pearson r = 0.905** (MSE 0.185× vs static baseline), confirming that KV-cache coefficients are highly predictable from input. Dynamic routing is viable; static compression is not.

---

## Planned Experiments

### Scale & Architecture Coverage

**Large model validation (Llama-3 / Mistral-7B–8B / Gemma-2 / Qwen2 / Phi-3)**
~~Apply GABE to attention and FFN groups in modern decoder-only transformers.~~ **Partially complete (Exp 32):** Llama 3 8B geometry confirmed — B1/B2 ratios 4–6.5×, B₃ ≈ 1.09× (random), float64 SVD exact at D=58.7M. Remaining: full 7-group sweep (`q/k/v/o/gate/up/down`), PEFT fine-tuning comparison at scale, and validation on Mistral/Gemma/Qwen/Phi families.

**GABE on large real datasets (10k–200k+ examples)**
Full benchmark comparison of GABE_FT vs LoRA / QLoRA / Full FT on standard NLP tasks (GLUE, MMLU, instruction following) and vision tasks (ImageNet, COCO). Report memory, throughput, and quality jointly. Current fine-tuning validation uses only 200–3000 training samples.

**Cross-model W̄ and B comparison — data-invariant vs data-specific patterns**
Compare W̄ and B between same-architecture models trained on different datasets (e.g., two Llama-3 8B models fine-tuned on different domains). Goal: identify which components are data-invariant (stable across training corpora) vs data-specific (domain fingerprints).

**Cross-architecture W̄ and B comparison — architecture vs data effect**
Compare W̄ and B between different architectures (Llama-3 vs Mistral vs Gemma-2) trained on identical data. Goal: disentangle the contribution of architecture topology vs training distribution to the affine weight space structure.

**GABE transplantation — intra-family and cross-architecture**
Transfer W̄, B, and α between models of the same family (e.g., Llama-3 8B → Llama-3 70B via projection) and cross-architecturally (Llama ↔ Mistral). Test whether functional skill transfer is possible via coefficient copying without re-training.

### Basis Dynamics & Freezing

**Freezing point determination**
Systematically measure: at what training step does updating $B$ stop being useful or begin degrading performance? Track subspace alignment SA(B_t, B_final) and validation loss jointly. Identify the "freezing point" after which $B$ can be locked with zero cost.

**Partial B unfreezing — curriculum and selective strategies**
Test: (a) unfreezing only the top-1 or top-2 singular vectors of $B$; (b) curriculum unfreezing (lock B early, release partially later); (c) selective unfreezing based on per-layer spectral percentile. Compare against fully frozen and fully trainable B.

**B stability and reuse without re-training**
Characterize "ideal" B extracted from large pretrained models. Test whether B extracted from one checkpoint can be reused directly for fine-tuning on a new task — without any B update. Measure how much task-specific performance is recoverable from α-only adaptation using a fixed pretrained B.

### Training & Optimization

**Joint GABE + head training**
Train $\overline{W}$, $\alpha$, and the task head (lm_head / classification head) simultaneously from the start, rather than sequentially. Measure interaction effects between head adaptation and W̄ adaptation.

**WBAR_ONLY mode**
Train only $\overline{W}$, freezing both $B_k$ and $\alpha$. Motivated by Exp 29, FT-CV, and FT-LM findings that α drift is ~250× smaller than W̄ drift during fine-tuning. Expected to yield 4× compression in l1 groups (vs 3× for GABE_FT).

**Dynamic router for α generation**
Train small networks to predict $\alpha$ from input $x$ or hidden states $h$. Extend Exp 5 and Exp 30 (Part C, r=0.905) to real-world NLP tasks with transformer backbones. Test transformer-based and hypernetwork-based routers vs the simple MLP baseline.

**Hybrid methods: GABE + LoRA / QLoRA / DoRA / Quantization**
Combine GABE basis freezing with low-rank $\overline{W}$ updates (GABE+LoRA), quantized $B_k$ (GABE+QLoRA), and weight-decomposed adaptation (GABE+DoRA). Profile the Pareto frontier of memory vs quality.

### Continual & Multi-Task Learning

**Advanced continual learning with GABE**
Extend Exp 21 (currently at chance accuracy) with proper baselines: linear probe, last-layer FT, LoRA rank-3. Increase per-task training budget and K to move past chance. Target: zero forgetting + above-chance accuracy on all tasks simultaneously.

**Multi-task training with task-conditioned α**
Train a single shared (W̄, B) and learn task-specific α sets for multiple tasks simultaneously. Compare against separate fine-tuned models and multi-task LoRA.

### Structural Analysis

**ATen computation graph influence on B shape and stability (Exp 26b extension)**
Exp 26b used 4 architecture variants with n_seeds=3. Extend to: (a) n_seeds ≥ 10 for reliable estimation; (b) more diverse op-chain variants including attention, layer norm, and gating; (c) test whether the seed-dominance result holds at 7B+ parameter scale.

**B₃ phenomenon — causal mechanism**
Systematic per-$B_k$ Rayleigh quotient breakdown across architectures. Scree plots of inter-layer singular values. Test whether VGG-11 (no skip connections) shows a B₃ drop — the key falsifiable prediction of the skip-connection topology hypothesis. N_grad ablation to rule out Fisher rank limitation as the cause.

**Tucker-GABE: cross-shape grouping**
Apply Tucker decomposition to project layers of different shapes into a shared kernel space (Exp 31). Determine the minimum Tucker rank $r$ for Rayleigh alignment preservation ≥ 90% across all ResNet-18 groups. Validate on transformer attention groups with varying head dimensions.

**KV-cache compression with dynamic routing**
Revisit Exp 30 Part B with dynamic α prediction (router r=0.905 confirmed). Test whether online α prediction can compensate for the static compression quality loss, targeting PPL degradation < 10%.

### Safety & Robustness

**GABE's effect on alignment, safety, and adversarial robustness**
Measure whether fine-tuning in GABE weight-space (frozen B) preserves safety behaviors better than FULL_FT. Test adversarial robustness of GABE_FT models vs FULL_FT models on standard attacks. Investigate whether B contains interpretable safety-relevant directions.

---

## Summary Statistics

| Property | Value | Source |
|----------|:-----:|--------|
| B₁, B₂ Rayleigh percentile (H / F / GCM) | **100th / 100th** | Exp 12 |
| B₃ Rayleigh percentile | ~35th (random) | Exp 12 |
| Spectral elevation at D=147k | **26.69×** above random | Exp 15 |
| B1 ratio on Llama 3 8B q_proj (D=16.7M) | **6.50×** above random | Exp 32 |
| B2 ratio on Llama 3 8B up_proj (D=58.7M) | **5.43×** above random | Exp 32 |
| B3 ratio on Llama 3 8B up_proj | **1.09×** (statistical noise) | Exp 32 |
| float64 SVD recon error at D=58.7M | **6.1e-14** (machine zero) | Exp 32 |
| span(B) alignment after fine-tuning | **0.9996** | Exp 19 |
| Same-seed cross-arch span(B) alignment | **2387×** above random | Exp 26b |
| α fragility vs W̄ (ε₅₀ ratio) | **4×** | Exp 20b |
| GABE_FT vs FULL_FT KL (ResNet-18 4-shot) | **0.000002** | FT-CV |
| GPT-2 GABE_FT PPL vs FULL_FT PPL | **116.47 vs 646.40** | FT-LM |
| Training VRAM reduction (GPT-2) | **12.4×** | FT-LM |
| Lossless conversion SA after re-extraction | **1.000000** | FT-LM2 |
| α prediction from Q (Pearson r) | **0.905** | Exp 30 |

---

## Citation

```bibtex
@misc{feklin2026gabe,
  title  = {GABE: Groupwise Affine Basis Encoding — Neural Networks as Memory-Addressed Systems},
  author = {Feklin, Dmitry},
  year   = {2026},
  url    = {https://github.com/FekDN/GABE}
}
```

**License:** Apache 2.0
