"""
Export helpers: Save and load basis matrices, checkpoints, and evaluation metrics.
"""

from typing import Dict, Any
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_basis_matrices(basis_dict: Dict[int, torch.Tensor], filepath: str):
    """Saves layer-wise basis matrices dictionary to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert keys to int if necessary
    clean_dict = {int(k): v.cpu() for k, v in basis_dict.items()}
    torch.save(clean_dict, filepath)
    print(f"✅ Saved basis matrices for layers {list(clean_dict.keys())} to {filepath}")


def load_basis_matrices(filepath: str, device: str = "cpu") -> Dict[int, torch.Tensor]:
    """Loads layer-wise basis matrices from disk."""
    loaded = torch.load(filepath, map_location=device)
    return {int(k): v for k, v in loaded.items()}


def export_safetensors_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
):
    """Saves the weight-edited model and tokenizer to safetensors directory."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Exported Safetensors checkpoint to {output_dir}")
