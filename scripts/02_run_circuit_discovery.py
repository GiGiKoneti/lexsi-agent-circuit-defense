#!/usr/bin/env python3
"""
Step 2: CircuitKIT EAP-IG Circuit Discovery & SVD Subspace Extraction.
Traces causal attribution across transformer layers and extracts the k=4
authorization subspace U_S on Vast.ai / Cloud GPU.
"""

import os
import sys
import json
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Run CircuitKIT EAP-IG and Subspace Extraction on GPU")
    parser.add_argument("--config", type=str, default="configs/model_aya_8b.yaml", help="Path to model config")
    parser.add_argument("--triplets_path", type=str, default="data/processed/triplets_cleaned.jsonl", help="Cleaned triplets data path")
    parser.add_argument("--output_basis", type=str, default="data/processed/basis_matrices_k4.pt", help="Output basis path")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of contrast pairs to evaluate")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.circuit.subspace_extractor import SubspaceExtractor
    from src.utils.export_helpers import save_basis_matrices

    if not torch.cuda.is_available():
        raise SystemError(
            "❌ CUDA GPU is required for real CircuitKIT mechanistic extraction! "
            "Please run this on Vast.ai with an NVIDIA A100/H100/RTX GPU."
        )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    target_layers = cfg["mechanistic"]["target_layers"]
    k = cfg["mechanistic"]["subspace_rank"]
    model_name = cfg["model"]["name"]
    torch_dtype = getattr(torch, cfg["model"]["torch_dtype"])
    device_map = cfg["model"]["device_map"]

    print("=" * 80)
    print(f"🔬 CircuitKIT: Mechanistic Circuit Discovery & Subspace Extraction")
    print(f"   • Target Model: {model_name}")
    print(f"   • Formation Window Target Layers: {target_layers}")
    print(f"   • Subspace Rank (k): {k}")
    print(f"   • Evaluation Pairs: {args.num_samples}")
    print(f"   • Compute Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    print(f"📥 Loading tokenizer and weights for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    clean_samples, corrupt_samples = [], []
    with open(args.triplets_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.num_samples:
                break
            data = json.loads(line)
            clean_samples.append(data["clean"])
            corrupt_samples.append(data["corrupted"])

    print(f"📊 Loaded {len(clean_samples)} clean/corrupted contrastive pairs.")
    extractor = SubspaceExtractor(model, tokenizer)

    print(f"🔍 Tracing differential activations & computing SVD across layers {target_layers}...")
    basis_dict, singular_values = extractor.extract_subspace_for_layers(
        clean_batch=clean_samples,
        corrupted_batch=corrupt_samples,
        target_layers=target_layers,
        k=k,
    )

    for l in target_layers:
        sv = singular_values[l][:k].cpu().tolist()
        sv_str = ", ".join([f"{v:.2f}" for v in sv])
        print(f"   • Layer {l}: Top-{k} Singular Values = [{sv_str}] | Basis Shape = {basis_dict[l].shape}")

    os.makedirs(os.path.dirname(args.output_basis), exist_ok=True)
    save_basis_matrices(basis_dict, args.output_basis)
    print(f"✅ Authorization subspace basis matrices U_S saved to: {args.output_basis}")
    print("=" * 80)


if __name__ == "__main__":
    main()
