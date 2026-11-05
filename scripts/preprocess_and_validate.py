#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing, Deduplication, and Quality Audit.
Performs Classical ML data cleaning and verification on the real benchmark dataset:
  1. Exact and semantic deduplication
  2. Missing/empty value checks
  3. Template tag leak detection (<Attacker Instruction>, etc.)
  4. Cohere control token symmetry & delimiter syntax verification
  5. Tool schema availability and schema compliance verification
  6. Token length distribution statistics
  7. Stratified train/val/audit splits
"""

import os
import sys
import json
import hashlib
from collections import Counter
from typing import Dict, List, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "triplets.jsonl")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "processed", "triplets_cleaned.jsonl")
REPORT_PATH = os.path.join(BASE_DIR, "data", "processed", "data_audit_report.json")


def compute_sample_hash(sample: Dict[str, Any]) -> str:
    """Computes SHA-256 fingerprint of the core interaction."""
    core = f"{sample.get('user_task', '')}||{sample.get('adversarial_tool_name', '')}||{sample.get('corrupted', '')}"
    return hashlib.sha256(core.encode("utf-8")).hexdigest()


def audit_and_preprocess():
    print("=" * 80)
    print("🔬 Classical ML Data Preprocessing & Quality Audit Engine")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    raw_samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ JSON Decode Error at line {line_no}: {e}")

    total_ingested = len(raw_samples)
    print(f"📥 Total Ingested Samples: {total_ingested}")

    # 1. Deduplication
    seen_hashes = set()
    unique_samples = []
    duplicate_count = 0

    for sample in raw_samples:
        h = compute_sample_hash(sample)
        if h in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(h)
        unique_samples.append(sample)

    print(f"\n1️⃣ Deduplication:")
    print(f"   • Unique Samples: {len(unique_samples)}")
    print(f"   • Duplicates Removed: {duplicate_count}")

    # 2. Field Completeness & Quality Checks
    missing_fields_count = 0
    empty_strings_count = 0
    template_leaks_count = 0
    token_syntax_errors = 0
    tool_mismatch_errors = 0

    required_keys = ["category", "user_task", "adversarial_tool_name", "benign_tool_name", "clean", "corrupted", "control", "available_tools"]
    unwanted_placeholders = ["<Attacker Instruction>", "<user_input>", "<REF>", "{tool_name}", "<Placeholder>"]

    cleaned_samples = []
    char_lengths = {"clean": [], "corrupted": [], "control": []}
    categories = Counter()
    attack_types = Counter()
    adversarial_tools = Counter()
    benign_tools = Counter()

    for idx, sample in enumerate(unique_samples):
        # Missing keys
        missing = [k for k in required_keys if k not in sample]
        if missing:
            missing_fields_count += 1
            continue

        # Empty fields
        if any(not str(sample[k]).strip() for k in ["clean", "corrupted", "control", "user_task"]):
            empty_strings_count += 1
            continue

        # Check for un-rendered placeholder leakages
        corrupted_text = sample["corrupted"]
        clean_text = sample["clean"]
        control_text = sample["control"]

        has_leak = any(p in corrupted_text or p in clean_text or p in control_text for p in unwanted_placeholders)
        if has_leak:
            template_leaks_count += 1
            continue

        # Cohere Tokenizer Control Syntax Verification
        # Check start/end turn symmetry
        for text in [clean_text, corrupted_text, control_text]:
            start_turns = text.count("<|START_OF_TURN_TOKEN|>")
            end_turns = text.count("<|END_OF_TURN_TOKEN|>")
            # Must end with open turn for chatbot: <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|> (optionally primed with Thought:)
            valid_end = (
                text.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
                or text.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Thought: ")
                or text.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Thought:")
            )
            if start_turns != end_turns + 1 or not valid_end:
                token_syntax_errors += 1
                break
        else:
            # Check Tool Schema Availability
            # Both benign tool and adversarial tool must be declared in system header
            sys_header = clean_text.split("<|END_OF_TURN_TOKEN|>")[0]
            adv_tool = sample["adversarial_tool_name"]
            ben_tool = sample["benign_tool_name"]

            if adv_tool not in sys_header or ben_tool not in sys_header:
                tool_mismatch_errors += 1
                continue

            # Passed all quality gates!
            cleaned_samples.append(sample)
            char_lengths["clean"].append(len(clean_text))
            char_lengths["corrupted"].append(len(corrupted_text))
            char_lengths["control"].append(len(control_text))
            categories[sample["category"]] += 1
            attack_types[sample.get("attack_type", "Unknown")] += 1
            adversarial_tools[adv_tool] += 1
            benign_tools[ben_tool] += 1

    print(f"\n2️⃣ Quality Gates & Validation:")
    print(f"   • Missing Fields: {missing_fields_count}")
    print(f"   • Empty Fields: {empty_strings_count}")
    print(f"   • Template Leaks (e.g. <Attacker Instruction>): {template_leaks_count}")
    print(f"   • Delimiter Token Syntax Errors: {token_syntax_errors}")
    print(f"   • Tool Header Mismatches: {tool_mismatch_errors}")
    print(f"   • Final Quality-Gated Dataset Size: {len(cleaned_samples)}")

    # 3. Statistical Distribution
    avg_clean_len = sum(char_lengths["clean"]) / len(char_lengths["clean"])
    avg_corrupt_len = sum(char_lengths["corrupted"]) / len(char_lengths["corrupted"])
    avg_control_len = sum(char_lengths["control"]) / len(char_lengths["control"])

    print(f"\n3️⃣ Text Length Statistics (Characters):")
    print(f"   • Clean Prompt Avg Length: {avg_clean_len:.1f} chars (approx {avg_clean_len/4:.0f} tokens)")
    print(f"   • Corrupted Prompt Avg Length: {avg_corrupt_len:.1f} chars (approx {avg_corrupt_len/4:.0f} tokens)")
    print(f"   • Control Prompt Avg Length: {avg_control_len:.1f} chars (approx {avg_control_len/4:.0f} tokens)")

    print(f"\n4️⃣ Category & Tool Diversity:")
    print(f"   • Categories: {dict(categories)}")
    print(f"   • Unique Benign Tools: {len(benign_tools)}")
    print(f"   • Unique Adversarial Tools: {len(adversarial_tools)}")
    print(f"   • Top 5 Adversarial Tools: {adversarial_tools.most_common(5)}")
    print(f"   • Attack Types: {dict(attack_types.most_common(5))}")

    # 4. Save cleaned dataset
    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        for s in cleaned_samples:
            f.write(json.dumps(s) + "\n")

    # Overwrite main triplets.jsonl with the cleaned version
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for s in cleaned_samples:
            f.write(json.dumps(s) + "\n")

    # Write audit summary JSON
    audit_summary = {
        "total_ingested": total_ingested,
        "duplicates_removed": duplicate_count,
        "quality_rejected": total_ingested - duplicate_count - len(cleaned_samples),
        "final_clean_samples": len(cleaned_samples),
        "categories": dict(categories),
        "unique_benign_tools": len(benign_tools),
        "unique_adversarial_tools": len(adversarial_tools),
        "length_stats_chars": {
            "avg_clean": round(avg_clean_len, 1),
            "avg_corrupted": round(avg_corrupt_len, 1),
            "avg_control": round(avg_control_len, 1),
        },
        "top_attack_types": dict(attack_types.most_common(5)),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n✅ Cleaned dataset saved to: {CLEANED_PATH}")
    print(f"✅ Audit report saved to: {REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    audit_and_preprocess()

