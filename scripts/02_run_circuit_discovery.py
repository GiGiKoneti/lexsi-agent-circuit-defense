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
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.circuit.eap_ig_tracer import CircuitDiscoveryEAPIG
from src.circuit.subspace_extractor import SubspaceExtractor
from src.utils.export_helpers import save_basis_matrices


def main():
    parser = argparse.ArgumentParser(description="Run CircuitKIT EAP-IG and Subspace Extraction")
    parser.add_argument("--config", type=str, default="configs/model_aya_8b.yaml", help="Path to model config")
    parser.add_argument("--triplets_path", type=str, default="data/processed/triplets.jsonl", help="Triplets data path")
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

    if args.mock or not torch.cuda.is_available():
        print("⚠️  Running in synthetic mock/CPU mode (for local testing)...")
        # Generate synthetic orthonormal basis matrices for target layers
        basis_dict = {}
        for l in target_layers:
            # Generate random orthogonal matrix via QR
            M = torch.randn(hidden_size, k)
            Q, _ = torch.linalg.qr(M)
            basis_dict[l] = Q.contiguous()
            print(f"   • Layer {l}: Extracted synthetic basis matrix shape: {basis_dict[l].shape}")

        save_basis_matrices(basis_dict, args.output_basis)
        print(f"✅ Mock circuit discovery complete! Basis saved to: {args.output_basis}")
        return

    # Real GPU execution
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

