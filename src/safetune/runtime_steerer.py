"""
SafeTune: Dynamically Gated Contrastive Activation Addition (CAA).
Provides optional runtime inference-time steering complement to orthogonal weight surgery.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class GatedCAASteerer:
    """
    Dynamically gated Contrastive Activation Addition wrapper.
    Monitors the pre-action residual state at a specified layer using a linear probe.
    If probe probability exceeds threshold tau, applies steering vector v_steer.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        target_layer: int = 20,
        threshold: float = 0.65,
        alpha: float = 1.2,
    ):
        self.model = model
        self.target_layer = target_layer
        self.threshold = threshold
        self.alpha = alpha
        self.hook_handle = None

    def register_steerer(
        self,
        steering_vector: torch.Tensor,
        gate_probe: Optional[nn.Linear] = None,
    ):
        """
        Attaches the forward pre-hook for dynamically gated steering.
        """
        device = next(self.model.parameters()).device
        v_norm = (steering_vector / torch.norm(steering_vector)).to(device)

        if gate_probe is None:
            # Fallback identity probe if none provided
            gate_probe = nn.Linear(self.model.config.hidden_size, 1).to(device)
            # Initialize with small positive bias for testing
            with torch.no_grad():
                gate_probe.weight.fill_(0.0)
                gate_probe.bias.fill_(1.0)
        else:
            gate_probe = gate_probe.to(device)

        gate_probe.eval()

        def gated_steering_hook(module, args, output):
            hidden_states = output[0] if isinstance(out := output, tuple) else output
            pre_action_state = hidden_states[:, -1, :]

            with torch.no_grad():
                gate_logits = gate_probe(pre_action_state.float())
                gate_prob = torch.sigmoid(gate_logits).squeeze(-1)

            for b in range(hidden_states.size(0)):
                prob = gate_prob[b].item() if gate_prob.dim() > 0 else gate_prob.item()
                if prob > self.threshold:
                    effective_alpha = self.alpha * ((prob - self.threshold) / (1.0 - self.threshold))
                    hidden_states[b, -1, :] += effective_alpha * v_norm.to(hidden_states.dtype)

            return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

        target_module = self._get_mlp_module(self.target_layer)
        self.hook_handle = target_module.register_forward_hook(gated_steering_hook)
        return self.hook_handle

    def remove(self):
        """Removes the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _get_mlp_module(self, layer_idx: int) -> nn.Module:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx].mlp
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx].mlp
        else:
            raise AttributeError(f"Unsupported model architecture: {type(self.model)}")
