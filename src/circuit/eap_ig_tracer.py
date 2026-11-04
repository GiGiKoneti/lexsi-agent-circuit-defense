"""
CircuitKIT: Edge Attribution Patching with Integrated Gradients (EAP-IG).
Locates the causal IPI transmission circuit and isolates authorization subspace via SVD.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class CircuitDiscoveryEAPIG:
    """
    Edge Attribution Patching with Integrated Gradients across transformer layers.
    Identifies the 'Formation Window' where MLP writes switch the model toward the malicious action token.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.d_model = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers

    def compute_eap_ig_edge_scores(
        self,
        clean_text: str,
        corrupted_text: str,
        target_tokens: Tuple[int, int],
        steps: int = 10,
        pos: int = -1,
    ) -> torch.Tensor:
        """
        Computes layer-by-layer attribution scores using Gauss-Legendre or linear path interpolation.
        target_tokens: (clean_token_id, adv_token_id)
        """
        self.model.eval()
        clean_ids = self.tokenizer(clean_text, return_tensors="pt")["input_ids"].to(self.device)
        corrupt_ids = self.tokenizer(corrupted_text, return_tensors="pt")["input_ids"].to(self.device)

        clean_token_id, adv_token_id = target_tokens

        # 1. Forward pass on clean input and cache activations
        clean_acts = {}

        def clean_hook(l):
            def fn(module, inp, out):
                val = out[0] if isinstance(out, tuple) else out
                clean_acts[l] = val[:, pos, :].detach().clone()
            return fn

        hooks = []
        for l in range(self.num_layers):
            mlp_module = self._get_mlp_module(l)
            hooks.append(mlp_module.register_forward_hook(clean_hook(l)))

        with torch.no_grad():
            self.model(clean_ids)

        for h in hooks:
            h.remove()

        # 2. Forward pass on corrupted input and cache activations
        corrupt_acts = {}

        def corrupt_hook(l):
            def fn(module, inp, out):
                val = out[0] if isinstance(out, tuple) else out
                corrupt_acts[l] = val[:, pos, :].detach().clone()
            return fn

        hooks = []
        for l in range(self.num_layers):
            mlp_module = self._get_mlp_module(l)
            hooks.append(mlp_module.register_forward_hook(corrupt_hook(l)))

        with torch.no_grad():
            self.model(corrupt_ids)

        for h in hooks:
            h.remove()

        # 3. Compute Integrated Gradients across M interpolation steps
        accumulated_grads = {l: torch.zeros_like(clean_acts[l]) for l in range(self.num_layers)}
        diff_acts = {l: corrupt_acts[l] - clean_acts[l] for l in range(self.num_layers)}

        for step_idx in range(1, steps + 1):
            alpha = step_idx / steps
            interpolated_acts = {
                l: (clean_acts[l] + alpha * diff_acts[l]).requires_grad_(True)
                for l in range(self.num_layers)
            }

            def inject_hook(l):
                def fn(module, inp, out):
                    out_mod = out[0].clone() if isinstance(out, tuple) else out.clone()
                    out_mod[:, pos, :] = interpolated_acts[l]
                    return (out_mod,) + out[1:] if isinstance(out, tuple) else out_mod
                return fn

            current_hooks = []
            for l in range(self.num_layers):
                mlp_module = self._get_mlp_module(l)
                current_hooks.append(mlp_module.register_forward_hook(inject_hook(l)))

            logits = self.model(corrupt_ids).logits[:, pos, :]
            # Metric: difference between adversarial target and clean target logit
            metric = logits[:, adv_token_id] - logits[:, clean_token_id]
            metric.backward(retain_graph=False)

            for l in range(self.num_layers):
                if interpolated_acts[l].grad is not None:
                    accumulated_grads[l] += interpolated_acts[l].grad.detach()

            for h in current_hooks:
                h.remove()

            self.model.zero_grad()

        # 4. Integrate attributions across layers
        layer_attributions = torch.zeros(self.num_layers)
        for l in range(self.num_layers):
            avg_grad = accumulated_grads[l] / steps
            eap_vector = diff_acts[l] * avg_grad
            layer_attributions[l] = eap_vector.abs().sum().item()

        return layer_attributions

    def _get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Retrieves the MLP submodule across standard transformer architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx].mlp
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx].mlp
        else:
            raise AttributeError(f"Unsupported model architecture for MLP hook extraction: {type(self.model)}")
