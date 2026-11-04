#!/usr/bin/env python3
"""
Step 1: CuratorKIT Triplet Synthesis
Generates clean, corrupted, and counterfactual control triplets for Cohere Aya models.
"""

import os
import sys
import json
import argparse

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.curator.triplet_generator import CohereAgentTripletCurator


def main():
    parser = argparse.ArgumentParser(description="Generate CuratorKIT contrastive triplets")
    parser.add_argument("--use_real_benchmarks", action="store_true", default=True, help="Load official InjecAgent benchmark data")
    parser.add_argument("--output_path", type=str, default="data/processed/triplets.jsonl", help="Output JSONL path")
    parser.add_argument("--curatorkit_output", type=str, default="data/processed/triplets_curatorkit.jsonl", help="CuratorKIT schema output path")
    args = parser.parse_args()

    print(f"🚀 Initializing CuratorKIT Engine with Lexsi-Labs CuratorKIT schema integration...")
    curator = CohereAgentTripletCurator()

    dh_base = "data/raw/injecagent/test_cases_dh_base.json"
    ds_base = "data/raw/injecagent/test_cases_ds_base.json"
    dh_enhanced = "data/raw/injecagent/test_cases_dh_enhanced.json"
    ds_enhanced = "data/raw/injecagent/test_cases_ds_enhanced.json"
    tools_path = "data/raw/injecagent/tools.json"

    if not (os.path.exists(dh_base) and os.path.exists(ds_base) and os.path.exists(tools_path)):
        raise FileNotFoundError(
            f"❌ Real benchmark files missing in data/raw/injecagent! "
            f"Run 'python scripts/download_real_benchmarks.py' first to download the official InjecAgent benchmark."
        )

    print(f"📥 Ingesting official InjecAgent benchmark datasets (UIUC Kang Lab)...")
    print(f"   • DH Base: {dh_base}")
    print(f"   • DS Base: {ds_base}")
    print(f"   • DH Enhanced: {dh_enhanced}")
    print(f"   • DS Enhanced: {ds_enhanced}")
    print(f"   • Tool Schemas: {tools_path}")

    triplets = curator.curate_from_injecagent(
        dh_path=dh_base,
        ds_path=ds_base,
        enhanced_dh_path=dh_enhanced,
        enhanced_ds_path=ds_enhanced,
        tools_path=tools_path,
        as_curatorkit_samples=False,
    )

    print(f"📋 Exporting official CuratorKIT DataSamples with ProvenanceRecords...")
    curatorkit_samples = curator.curate_from_injecagent(
        dh_path=dh_base,
        ds_path=ds_base,
        enhanced_dh_path=dh_enhanced,
        enhanced_ds_path=ds_enhanced,
        tools_path=tools_path,
        as_curatorkit_samples=True,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    with open(args.curatorkit_output, "w", encoding="utf-8") as f:
        for s in curatorkit_samples:
            if hasattr(s, "model_dump_json"):
                f.write(s.model_dump_json() + "\n")
            elif hasattr(s, "json"):
                f.write(s.json() + "\n")
            else:
                f.write(json.dumps(s) + "\n")

    dh_count = sum(1 for t in triplets if t["category"] == "DH")
    ds_count = sum(1 for t in triplets if t["category"] == "DS")
    print(f"✅ Curated {len(triplets)} real benchmark triplets successfully!")
    print(f"   • Direct Harm (DH): {dh_count}")
    print(f"   • Data Stealing (DS): {ds_count}")
    print(f"   • Saved raw triplets to: {args.output_path}")
    print(f"   • Saved CuratorKIT DataSamples to: {args.curatorkit_output}")


if __name__ == "__main__":
    main()
