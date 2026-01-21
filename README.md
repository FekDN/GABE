# Method: Groupwise Affine Basis Encoding (GABE)

## 1. Problem Statement

Given a group of weight tensors of identical shape from a neural network,

$$
\mathcal{W} = \{ W_1, W_2, \dots, W_N \}, \quad W_i \in \mathbb{R}^{d_1 \times \dots \times d_m}
$$

the objective is to find a representation that:

1.  **Exactly** reconstructs all $W_i$.
2.  Utilizes a **shared structural basis** for the entire group.
3.  **Minimizes** the required storage volume.

---

## 2. Core Proposition

**Theorem (Affine Dimensionality of a Tensor Group).**
The set $\mathcal{W}$ lies in an affine subspace of dimension at most $N-1$.

### Proof

Consider the group's centroid (the mean tensor):

$$
\bar W = \frac{1}{N} \sum_{i=1}^N W_i
$$

Define the centered tensors (deviations from the centroid):

$$
\Delta W_i = W_i - \bar W
$$

By construction, the sum of these deviations is zero:

$$
\sum_{i=1}^N \Delta W_i = 0
$$

Consequently, the set of vectors $\{ \Delta W_i \}$ is linearly dependent, and the dimension of their linear span, $\mathrm{span}(\{ \Delta W_i \})$, does not exceed $N-1$. Since each $W_i$ is a translation of a vector $\Delta W_i$ by the same vector $\bar W$, the entire set $\mathcal{W}$ lies within the affine subspace $\bar W + \mathrm{span}(\{ \Delta W_i \})$ of the same dimension.

---

## 3. Corollary

There exists an orthonormal basis

$$
\mathcal{B} = \{ B_1, \dots, B_K \}, \quad K \le N-1
$$

and a set of coefficients $\alpha_{ik} \in \mathbb{R}$ such that:

$$
\forall i \in \{1, \dots, N\}:\quad
W_i = \bar W + \sum_{k=1}^{K} \alpha_{ik} B_k
$$

---

## 4. Basis Construction (Algorithm)

### Input

*   Tensors $W_1, \dots, W_N$

### Steps

1.  **Compute the centroid:** $\bar W = \frac{1}{N} \sum W_i$.
2.  **Form the deviation matrix:**

$$
    X =
    \begin{bmatrix}
    \mathrm{vec}(W_1 - \bar W)^\top \\
    \vdots \\
    \mathrm{vec}(W_N - \bar W)^\top
    \end{bmatrix}
    \in \mathbb{R}^{N \times D}
$$

    where $D = d_1 \times \dots \times d_m$.
4.  **Perform Singular Value Decomposition (SVD):**

$$
    X = U \Sigma V^\top
$$

5.  **Extract the basis and coefficients:**
    *   **Basis:** $B_k = \mathrm{reshape}(\text{k-th row of } V^\top)$, for $k=1,\dots,K$ .
    *   **Coefficients:** The matrix $\alpha = U \Sigma$ .

---

## 5. Representation and Reconstruction

**Stored Components:**

*   **Shared:** The centroid $\bar W$ and the basis $\{B_k\}_{k=1}^K$.
*   **Per-Tensor:** The coefficients $\{\alpha_{ik}\}$.

**Reconstruction Formula:**

$$
\hat W_i = \bar W + \sum_{k=1}^{K} \alpha_{ik} B_k
$$

When using the full rank $K=N-1$, the reconstruction is **numerically exact** (up to machine precision).

---

## 6. Component Compression (Practical Implementation)

It has been experimentally demonstrated that the components $\bar W$ and $\{B_k\}$ possess an internal low-rank structure, while the coefficients $\{\alpha_{ik}\}$ are of small magnitude. This enables their **lossless compression** via a "formula + residual" approach:

1.  **Compressing $\bar W$ and $B_k$**: Each of these tensors is represented as the sum of a low-rank approximation (the "formula") and a residual.

$$
    \bar W = \tilde{\bar W} + R_{\bar W}, \quad B_k = \tilde B_k + R_k
$$

2.  **Storage**:
    *   The **formulas** ($\tilde{\bar W}, \tilde B_k$) are stored via their compact SVD components (e.g., in `float16`).
    *   The **residuals** ($R_{\bar W}, R_k$) exhibit low energy and are efficiently compressed through aggressive quantization (e.g., to `int8`).
    *   The **coefficients** ($\alpha_{ik}$) are small in magnitude and can be stored directly (e.g., in `float16`).

---

## 7. Method Properties

| Property         | Description                                    |
| ---------------- | ---------------------------------------------- |
| **Type**         | Deterministic, factorization-based             |
| **Loss**         | Lossless (with full rank and no residual quantization) |
| **Architecture** | Agnostic                                       |
| **Inference**    | Unchanged (model is reconstructed before use)   |
| **Training**     | Not required                                   |
| **Applicability**| Any group of tensors with identical shape      |

---

## 8. Limitations

1.  Requires a group of two or more tensors of the same shape.
2.  Does not reduce the computational complexity (FLOPs) of inference.
3.  The compression ratio is dependent on the number of tensors in the group ($L$), following the approximation $\approx \frac{4L}{L-0.5}$.

---

## 9. Abstract

**Groupwise Affine Basis Encoding (GABE)** is a method for the exact representation of neural network weights, based on the finding that weights of tensors with identical topology lie in a shared affine subspace of dimension no greater than the number of tensors minus one. The method constructs an orthonormal basis for this subspace, encoding each tensor via a shared mean and a set of coefficients. This representation is numerically exact and allows for aggressive lossless compression of its components, without altering the model's architecture or inference process.

---

## 10. Potential Extensions

*   Entropy coding of quantized residuals and coefficients.
*   Joint optimization of the basis and quantization parameters.
*   Application to weight deltas for efficient fine-tuning.


# A Comparative Analysis of Weight Compression Methods
An empirical study of 22 weight tensor compression methods revealed universal principles across diverse neural network architectures (Transformer and Convolutional) and enabled the formulation of quantitatively justified recommendations.

---

## 11. Methodology

**Weight Grouping:**
Weight tensors were grouped by functional role and shape to isolate semantically homogeneous data. This strategy enabled the application of methods that exploit inter-layer correlations, such as Singular Value Decomposition (SVD).

**Evaluation Metrics:**

| Metric            | Formula                                            | Purpose                                                                                                     |
|-------------------|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| MSE               | `MSE = mean((W - W_hat)^2)`                        | Sensitive to large outliers; measures the mean squared error of the approximation.                          |
| RMSE              | `RMSE = sqrt(MSE)`                                 | Provides a direct measure of the average error in the same units as the weights.                            |
| MAE               | `MAE = mean(abs(W - W_hat))`                       | A robust measure of the average error, less sensitive to outliers.                                          |
| SNR               | `SNR(dB) = 10 * log10(variance(W) / MSE)`          | A logarithmic metric representing the ratio of signal power to error power; key for method selection. |
| Compression Ratio | `CR = Original Size / Compressed Size`             | Quantifies the degree of memory savings.                                                                    |

*Here, `W` represents the original weight tensor, and `W_hat` represents the reconstructed tensor.*

**Standardized Testing Procedure:**
All methods were evaluated on identical weight groups under consistent memory form-factor assumptions and using the same set of metrics.

---
![Test results](test.jpg)
---

## 12. Key Findings

### Principle 1: The Universality of Inter-Layer Redundancy

*   **Method:** SVD(float16), K = L−1
*   **Results:** SNR = 141–145 dB, CR = 2.0× across all weight groups in both GPT-2 and ResNet-18.
*   **Conclusion:** Inter-layer weight redundancy is a fundamental property of deep architectures. Full-rank SVD allows for its nearly lossless elimination.

**Implication:**
Any compression method that disregards inter-layer correlation will inevitably be suboptimal in terms of reconstruction quality.

---

### Principle 2: The Superiority of Adaptive Quantization

*   **Block-FP8:** Achieved an SNR 10–24 dB higher than standard int8 (e.g., for GPT-2 FFN2 layers: 42 dB vs. 17.9 dB).
*   **Grouped Quantization (GQ-int4):** Proved to be the only viable 4-bit method, yielding an SNR of ~20 dB, whereas naive int4 dropped to 0.5–8 dB.

**Conclusion:**
Adaptive, localized scaling is essential for low-bit quantization. Naive global quantization leads to a catastrophic loss of information.

---

### Principle 3: The Efficacy of Hybrid Hierarchical Approaches

*   **Method:** SVD(int8) + Delta(SVD, K=1)
*   **Results:**
    *   GPT-2: SNR ~32 dB, CR ~3×
    *   ResNet-18: SNR up to 38.5 dB
*   **Interpretation:** This approach first compresses the principal components (SVD) and then encodes the residuals (delta), preserving high fidelity at a significant compression rate.

**Conclusion:**
Hierarchical compression is a powerful tool for balancing reconstruction quality and compression ratio.

---

### Ineffective and Niche Strategies

| Method                          | Issue                                                                                                            |
|---------------------------------|------------------------------------------------------------------------------------------------------------------|
| Tucker, TT-SVD                  | Extreme compression (up to 287×) but with SNR < 2 dB; the weight structure does not fit the tensor decomposition model. |
| Naive Pruning (50%)             | Severe quality degradation; SVD components remain dense and informative.                                         |
| Sparse SVD (80%)                | Sparsification does not reduce error effectively; SVD components are dense and ill-suited for pruning.             |

Methods relying on extreme sparsity or theoretical tensor decompositions lose their utility as general-purpose compressors without adaptation to the statistical properties of the weights.

---

## 13. Universal Principles

1.  **Full-rank SVD with float16** serves as the quality baseline for what is losslessly achievable.
2.  **Adaptive quantization of local blocks** enables low-bit storage with acceptable fidelity.
3.  **Hybrid hierarchical schemes** (SVD + delta) provide the best CR/SNR trade-off, especially for mid-range compression.
4.  **Global low-bit methods, sparsity, and theoretical tensor decompositions** are niche strategies with limited applicability.

For practical weight compression, a **universal strategy** is:

1.  Perform full-rank SVD(float16) to establish the baseline accuracy.
2.  Employ adaptive quantization (Block-FP8 / GQ-int4) to reduce size while preserving SNR.
3.  Use hierarchical compression (SVD + Delta) for a balanced CR/SNR compromise.

---

## License

The source code of this project is licensed under the **Apache License 2.0**. A full copy of the license is available in the `LICENSE` file in the root directory of this repository and can also be viewed at [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

---
### Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
---
