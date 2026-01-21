# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.

import torch
from typing import List, Tuple, Dict, Any
import math

class GABE:
   
    def _extract_svd_components(self, weights_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        if not weights_list: raise ValueError("The scale list cannot be empty.")
        original_dtype = weights_list[0].dtype
        stacked = torch.stack(weights_list)
        original_shape = stacked.shape
        w_bar = torch.mean(stacked, dim=0)
        L = original_shape[0]
        if L <= 1:
            return w_bar, torch.empty(0, dtype=original_dtype), torch.empty(0, dtype=original_dtype), original_shape
        stacked_double = stacked.to(torch.float64)
        flattened_double = stacked_double.view(L, -1)
        mean_flat_double = w_bar.to(torch.float64).view(-1)
        centered = flattened_double - mean_flat_double
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        K = L - 1
        B_stacked = Vh[:K].view(K, *original_shape[1:]).to(original_dtype)
        coeffs = (U[:, :K] @ torch.diag(S[:K])).to(original_dtype)
        return w_bar, B_stacked, coeffs, original_shape

    def _compress_matrix(self, matrix: torch.Tensor, rank: int) -> Tuple[Tuple, torch.Tensor]:
        """A universal function for compressing any 2D/3D matrix using the 'formula + remainder' method."""
        is_batch = matrix.dim() == 3
        # SVD works with batches (3D) or single matrices (2D)
        U, S, Vh = torch.linalg.svd(matrix.to(torch.float64))
        if is_batch:
            U_r, S_r, Vh_r = U[..., :, :rank], S[..., :rank], Vh[..., :rank, :]
            formula = (U_r @ torch.diag_embed(S_r) @ Vh_r).to(matrix.dtype)
        else: # Single matrix
            U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
            formula = (U_r @ torch.diag(S_r) @ Vh_r).to(matrix.dtype)
        residual = matrix - formula
        formulas = (U_r.to(matrix.dtype), S_r.to(matrix.dtype), Vh_r.to(matrix.dtype))
        return formulas, residual

    def _decompress_matrix(self, formulas: Tuple, residuals: torch.Tensor) -> torch.Tensor:
        """Universal function for matrix restoration."""
        U_r, S_r, Vh_r = formulas
        is_batch = U_r.dim() == 3
        U_r_d, S_r_d, Vh_r_d = U_r.to(torch.float64), S_r.to(torch.float64), Vh_r.to(torch.float64)
        if is_batch:
            formula_double = U_r_d @ torch.diag_embed(S_r_d) @ Vh_r_d
        else:
            formula_double = U_r_d @ torch.diag(S_r_d) @ Vh_r_d
        return formula_double.to(residuals.dtype) + residuals

    def _reconstruct_weights(self, w_bar: torch.Tensor, B: torch.Tensor, coeffs: torch.Tensor, original_shape: tuple) -> List[torch.Tensor]:
        num_layers = original_shape[0]
        if num_layers <= 1: return [w_bar.clone()]
        K = B.shape[0]
        B_flat = B.view(K, -1)
        reconstructed_centered_flat = torch.matmul(coeffs, B_flat)
        reconstructed_flat = reconstructed_centered_flat + w_bar.view(1, -1)
        reconstructed_stacked = reconstructed_flat.view(original_shape[0], *original_shape[1:])
        return [t.clone() for t in reconstructed_stacked]

    def compress(self, weights_list: List[torch.Tensor], basis_rank: int = 1, w_bar_rank: int = 16) -> Dict[str, Any]:
        """Compresses a group of tensors, including w_bar."""
        w_bar_orig, B_orig, coeffs, shape = self._extract_svd_components(weights_list)
        w_bar_formulas, w_bar_residuals = self._compress_matrix(w_bar_orig, rank=w_bar_rank)
        if B_orig.numel() > 0:
            basis_formulas, basis_residuals = self._compress_matrix(B_orig, rank=basis_rank)
        else:
            basis_formulas, basis_residuals = (), torch.empty(0, device=B_orig.device, dtype=B_orig.dtype)
        return { "w_bar_formulas": w_bar_formulas, "w_bar_residuals": w_bar_residuals,
                 "coeffs": coeffs, "basis_formulas": basis_formulas, "basis_residuals": basis_residuals,
                 "original_shape": shape }

    def decompress(self, compressed_data: Dict[str, Any]) -> List[torch.Tensor]:
        """Recovers a group of tensors from compressed data."""
        w_bar_rec = self._decompress_matrix(
            compressed_data["w_bar_formulas"], compressed_data["w_bar_residuals"]
        )
        if compressed_data["basis_residuals"].numel() > 0:
            B_rec = self._decompress_matrix(
                compressed_data["basis_formulas"], compressed_data["basis_residuals"] )
        else:
            B_rec = torch.empty(0, device=w_bar_rec.device, dtype=w_bar_rec.dtype)
        return self._reconstruct_weights(
            w_bar_rec, B_rec, compressed_data["coeffs"], compressed_data["original_shape"] )