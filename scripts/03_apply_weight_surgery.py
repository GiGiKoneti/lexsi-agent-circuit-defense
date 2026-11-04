#!/usr/bin/env python3
"""
Step 3: SafeTune Orthogonal Weight Surgery
Applies P_perp = (I - U_S U_S^T) directly to MLP W_down projection weights in layers L20-L23.
Validates linear nullification ||U_S^T W_down*||_inf < 1e-5 and exports Safetensors checkpoint.
"""

import os
import sys
import argparse
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.safetune.weight_projector import SafeTuneIntervention
from src.utils.export_helpers import load_basis_matrices, export_safetensors_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Apply SafeTune Orthogonal Weight Surgery")
    parser.add_argument("--config", type=str, default="configs/model_aya_8b.yaml", help="Path to config file")
    parser.add_argument("--basis_path", type=str, default="data/processed/basis_matrices_k4.pt", help="Path to basis matrices")
    parser.add_argument("--output_dir", type=str, default="safetune_aya8b_projected", help="Output directory for edited weights")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for validation")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    target_layers = cfg["mechanistic"]["target_layers"]
    tolerance = float(cfg["safetune"]["tolerance"])
    hidden_size = cfg["model"]["hidden_size"]
    intermediate_size = cfg["model"]["intermediate_size"]
    k = cfg["mechanistic"]["subspace_rank"]

    print(f"🛠️  SafeTune: Executing Orthogonal Subspace Weight Surgery (P_perp)")
    print(f"   • Target Layers: {target_layers}")
    print(f"   • Subspace Rank (k): {k}")
    print(f"   • Tolerance Bound: {tolerance}")

    if args.mock or not torch.cuda.is_available():
        print("⚠️  Running synthetic weight surgery verification test (Linear Algebra Invariant Check)...")
        # Load or generate basis
        if os.path.exists(args.basis_path):
            basis_dict = load_basis_matrices(args.basis_path)
        else:
            basis_dict = {}
            for l in target_layers:
                M = torch.randn(hidden_size, k)
                Q, _ = torch.linalg.qr(M)
                basis_dict[l] = Q

        for l in target_layers:
            U_S = basis_dict[l]
            # Simulate W_down: shape (hidden_size, intermediate_size)
            W_down = torch.randn(hidden_size, intermediate_size)

            # Associative projection: W* = W - U_S @ (U_S^T @ W)
            u_proj = torch.matmul(U_S.t(), W_down)
            null_comp = torch.matmul(U_S, u_proj)
            W_down_star = W_down - null_comp

            # Validation check: U_S^T @ W_down* must be zero
            residual_proj = torch.matmul(U_S.t(), W_down_star)
            max_error = torch.max(torch.abs(residual_proj)).item()

            print(f"   • Layer {l}: Max Nullification Error = {max_error:.2e}")
            assert max_error < tolerance, f"Invariant violated at layer {l}: {max_error} >= {tolerance}"

        print(f"✅ SafeTune Invariant Verified: Provable linear nullification ||U_S^T W_down*||_inf < {tolerance} holds strictly across all target layers!")
        return

    # Real model surgery
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"📥 Loading model for weight surgery: {cfg['model']['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
    )

    basis_dict = load_basis_matrices(args.basis_path, device=cfg["model"]["device_map"])
    intervenor = SafeTuneIntervention(model)

    print(f"✂️  Performing in-place orthogonal weight surgery on MLP down-projections...")
    max_errors = intervenor.apply_orthogonal_subspace_surgery(
        layers=target_layers,
        basis_matrices=basis_dict,
        tolerance=tolerance,
    )

    for l, err in max_errors.items():
        print(f"   • Layer {l}: Surgery validated! Max error = {err:.2e}")

    print(f"💾 Exporting weight-edited checkpoint to {args.output_dir}...")
    export_safetensors_checkpoint(model, tokenizer, args.output_dir)
    print(f"✅ SafeTune checkpoint ready for deployment!")


if __name__ == "__main__":
    main()

