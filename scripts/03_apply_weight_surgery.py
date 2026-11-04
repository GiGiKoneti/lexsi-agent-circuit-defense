#!/usr/bin/env python3
"""
Step 3: SafeTune Orthogonal Weight Surgery
Executes permanent P_perp orthogonal projection on MLP W_down weights:
  W_down* = P_perp W_down = (I - U_S U_S^T) W_down = W_down - U_S (U_S^T W_down)
Zero added FLOPs, zero added parameters, zero added inference latency (Delta tau = 0.00 ms).
Exports safetensors checkpoint for deployment.
"""

import os
import sys
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Apply SafeTune Orthogonal Weight Surgery on Model Weights")
    parser.add_argument("--config", type=str, default="configs/model_aya_8b.yaml", help="Path to config file")
    parser.add_argument("--basis_path", type=str, default="data/processed/basis_matrices_k4.pt", help="Path to extracted basis matrices")
    parser.add_argument("--output_dir", type=str, default="safetune_aya8b_projected", help="Output directory for edited weights")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.safetune.weight_projector import SafeTuneIntervention
    from src.utils.export_helpers import load_basis_matrices, export_safetensors_checkpoint

    if not os.path.exists(args.basis_path):
        raise FileNotFoundError(
            f"❌ Basis file not found at '{args.basis_path}'. "
            f"Please run 'scripts/02_run_circuit_discovery.py' first on GPU to extract the authorization subspace."
        )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    target_layers = cfg["mechanistic"]["target_layers"]
    tolerance = float(cfg["safetune"]["tolerance"])
    k = cfg["mechanistic"]["subspace_rank"]
    model_name = cfg["model"]["name"]
    torch_dtype = getattr(torch, cfg["model"]["torch_dtype"])
    device_map = cfg["model"]["device_map"]

    print("=" * 80)
    print(f"🛠️  SafeTune: Executing Orthogonal Subspace Weight Surgery (P_perp)")
    print(f"   • Target Model: {model_name}")
    print(f"   • Target Layers: {target_layers}")
    print(f"   • Subspace Rank (k): {k}")
    print(f"   • Tolerance Bound: {tolerance}")
    print(f"   • Basis Source: {args.basis_path}")
    print("=" * 80)

    print(f"📥 Loading basis matrices from {args.basis_path}...")
    basis_dict = load_basis_matrices(args.basis_path)

    print(f"📥 Loading base model weights: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    intervenor = SafeTuneIntervention(model)

    print(f"✂️  Executing in-place orthogonal weight surgery on MLP W_down projection weights...")
    max_errors = intervenor.apply_orthogonal_subspace_surgery(
        layers=target_layers,
        basis_matrices=basis_dict,
        tolerance=tolerance,
    )

    for l, err in max_errors.items():
        print(f"   • Layer {l}: Surgery validated! Max residual error = {err:.2e} (bound: < {tolerance})")

    print(f"💾 Exporting weight-edited Safetensors checkpoint to: {args.output_dir}...")
    export_safetensors_checkpoint(model, tokenizer, args.output_dir)
    print(f"✅ SafeTune checkpoint successfully saved to {args.output_dir}!")
    print("=" * 80)


if __name__ == "__main__":
    main()
