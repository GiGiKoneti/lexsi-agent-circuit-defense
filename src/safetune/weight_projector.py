"""
SafeTune: Subspace Weight Surgery.
Executes permanent P_perp orthogonal projection on W_down weights:
  W_down* = P_perp W_down = (I - U_S U_S^T) W_down = W_down - U_S (U_S^T W_down)
Zero added FLOPs, zero added parameters, zero added inference latency (Delta tau = 0.00 ms).
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class SafeTuneIntervention:
    """
    Direct linear algebraic weight surgery on MLP down-projection weights.
    Provably nullifies communication from intermediate MLP representations to residual stream
    along the authorization subspace U_S.
    """

    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.d_model = model.config.hidden_size

    def apply_orthogonal_subspace_surgery(
        self,
        layers: List[int],
        basis_matrices: Dict[int, torch.Tensor],
        tolerance: float = 1e-5,
    ) -> Dict[int, float]:
        """
        Executes in-place orthogonal weight surgery on W_down matrices for given layers.
        Returns:
          max_errors: Dict mapping layer index to residual projection infinity norm.
        """
        max_errors = {}

        with torch.no_grad():
            for l in layers:
                if l not in basis_matrices:
                    continue

                down_proj = self._get_down_proj(l)
                device = down_proj.weight.data.device
                dtype = down_proj.weight.data.dtype

                U_S = basis_matrices[l].to(device=device, dtype=dtype)
                W = down_proj.weight.data

                # Associative projection: W* = W - U_S @ (U_S^T @ W)
                # Avoids O(d^2) memory footprint
                u_proj = torch.matmul(U_S.t(), W)
                null_component = torch.matmul(U_S, u_proj)
                W.sub_(null_component)

                # Validation assertion: confirm orthogonality to target basis
                residual_projection = torch.matmul(U_S.t(), down_proj.weight.data)
                max_error = torch.max(torch.abs(residual_projection)).item()
                max_errors[l] = max_error

                assert max_error < tolerance, (
                    f"Surgery failed at Layer {l}: Max error {max_error} exceeds tolerance {tolerance}."
                )

        return max_errors

    def _get_down_proj(self, layer_idx: int) -> nn.Linear:
        """Retrieves down_proj module from transformer architecture."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx].mlp.down_proj
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx].mlp.down_proj
        else:
            raise AttributeError(f"Unsupported model architecture: {type(self.model)}")
