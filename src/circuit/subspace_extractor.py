"""
CircuitKIT: SVD Subspace Extraction.
Extracts orthonormal basis U_S in R^{d x k} from differential residual activations across target layers.
"""

from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SubspaceExtractor:
    """
    Extracts the k-dimensional authorization subspace U_S from differential activations:
      Delta_x = x_corrupt - x_clean
    evaluated at the terminal pre-action decision token.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.d_model = model.config.hidden_size

    def extract_subspace_for_layers(
        self,
        clean_batch: List[str],
        corrupted_batch: List[str],
        target_layers: List[int],
        k: int = 4,
        pos: int = -1,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extracts basis U_S and singular values spectrum across all specified target layers.
        Returns:
          basis_matrices: Dict[layer_idx, torch.Tensor of shape (d_model, k)]
          singular_values: Dict[layer_idx, torch.Tensor of singular values]
        """
        self.model.eval()
        diff_cache: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

        for clean_text, corrupt_text in zip(clean_batch, corrupted_batch):
            c_ids = self.tokenizer(clean_text, return_tensors="pt")["input_ids"].to(self.device)
            i_ids = self.tokenizer(corrupt_text, return_tensors="pt")["input_ids"].to(self.device)

            clean_out, corrupt_out = {}, {}

            def get_act(storage, layer_idx):
                def fn(mod, inp, out):
                    val = out[0] if isinstance(out, tuple) else out
                    storage[layer_idx] = val[:, pos, :].detach().clone()
                return fn

            # 1. Capture clean activations
            hooks_clean = []
            for l in target_layers:
                mlp_mod = self._get_mlp_module(l)
                hooks_clean.append(mlp_mod.register_forward_hook(get_act(clean_out, l)))

            with torch.no_grad():
                self.model(c_ids)

            for h in hooks_clean:
                h.remove()

            # 2. Capture corrupted activations
            hooks_corrupt = []
            for l in target_layers:
                mlp_mod = self._get_mlp_module(l)
                hooks_corrupt.append(mlp_mod.register_forward_hook(get_act(corrupt_out, l)))

            with torch.no_grad():
                self.model(i_ids)

            for h in hooks_corrupt:
                h.remove()

            # 3. Store delta = corrupt - clean
            for l in target_layers:
                delta = (corrupt_out[l] - clean_out[l]).squeeze(0)  # Shape: (d_model,)
                diff_cache[l].append(delta)

        # 4. Compute SVD per target layer
        basis_matrices: Dict[int, torch.Tensor] = {}
        singular_values: Dict[int, torch.Tensor] = {}

        for l in target_layers:
            diff_matrix = torch.stack(diff_cache[l], dim=1)  # Shape: (d_model, num_samples)
            U, S, Vh = torch.linalg.svd(diff_matrix.float(), full_matrices=False)
            U_S = U[:, :k].contiguous()  # Shape: (d_model, k)
            basis_matrices[l] = U_S
            singular_values[l] = S

        return basis_matrices, singular_values

    def _get_mlp_module(self, layer_idx: int):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx].mlp
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx].mlp
        else:
            raise AttributeError(f"Unsupported model architecture: {type(self.model)}")
