#!/usr/bin/env python3
"""
Step 2: CircuitKIT EAP-IG Circuit Discovery & SVD Subspace Extraction.
Traces attribution across layers and extracts the k=4 authorization subspace U_S.
Supports full GPU execution as well as mock/dry-run mode for local validation.
"""

import os
import sys
import json
import argparse
import yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Run CircuitKIT EAP-IG and Subspace Extraction")
    parser.add_argument("--config", type=str, default="configs/model_aya_8b.yaml", help="Path to model config")
    parser.add_argument("--triplets_path", type=str, default="data/processed/triplets_cleaned.jsonl", help="Triplets data path")
    parser.add_argument("--output_basis", type=str, default="data/processed/basis_matrices_k4.pt", help="Output basis path")
    parser.add_argument("--mock", action="store_true", help="Run in synthetic mock mode for local validation without GPU")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of pairs to evaluate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    target_layers = cfg["mechanistic"]["target_layers"]
    k = cfg["mechanistic"]["subspace_rank"]
    hidden_size = cfg["model"]["hidden_size"]

    print(f"🔬 CircuitKIT: Running Circuit Discovery & Authorization Subspace Extraction")
    print(f"   • Model: {cfg['model']['name']}")
    print(f"   • Target Formation Window Layers: {target_layers}")
    print(f"   • Subspace Rank (k): {k}")

    if args.mock or not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("⚠️  Running in synthetic mock/CPU mode (for local testing)...")
        # Generate synthetic orthonormal basis matrices for target layers using QR
        np.random.seed(42)
        basis_dict = {}
        for l in target_layers:
            M = np.random.randn(hidden_size, k)
            Q, _ = np.linalg.qr(M)
            basis_dict[l] = Q
            print(f"   • Layer {l}: Extracted synthetic basis matrix shape: {basis_dict[l].shape}")

        os.makedirs(os.path.dirname(args.output_basis), exist_ok=True)
        if TORCH_AVAILABLE:
            torch_dict = {l: torch.from_numpy(v).float() for l, v in basis_dict.items()}
            from src.utils.export_helpers import save_basis_matrices
            save_basis_matrices(torch_dict, args.output_basis)
        else:
            # Save numpy archive as fallback for CPU environment without PyTorch
            npz_path = args.output_basis.replace(".pt", ".npz")
            np.savez(npz_path, **{str(l): v for l, v in basis_dict.items()})
            print(f"   • Saved CPU-mock basis matrices to: {npz_path}")

        print(f"✅ Mock circuit discovery complete!")
        return

    # Real GPU execution
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.circuit.subspace_extractor import SubspaceExtractor
    from src.utils.export_helpers import save_basis_matrices
    print(f"📥 Loading model and tokenizer: {cfg['model']['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
    )

    clean_samples, corrupt_samples = [], []
    with open(args.triplets_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= args.num_samples:
                break
            data = json.loads(line)
            clean_samples.append(data["clean"])
            corrupt_samples.append(data["corrupted"])

    extractor = SubspaceExtractor(model, tokenizer)
    print(f"🔍 Extracting authorization subspace U_S across layers {target_layers} using SVD...")
    basis_dict, singular_values = extractor.extract_subspace_for_layers(
        clean_batch=clean_samples,
        corrupted_batch=corrupt_samples,
        target_layers=target_layers,
        k=k,
    )

    save_basis_matrices(basis_dict, args.output_basis)
    print(f"✅ Extracted and saved basis matrices to {args.output_basis}")


if __name__ == "__main__":
    main()

