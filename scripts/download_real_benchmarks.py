#!/usr/bin/env python3
"""
Downloads official, published Indirect Prompt Injection datasets directly from:
  1. InjecAgent (uiuc-kang-lab/InjecAgent)
  2. BIPIA (microsoft/BIPIA)
  3. XSTest (paul-rottger/commonsense-evaluation)
"""

import os
import sys
import urllib.request
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


URLS = {
    "injecagent": {
        "test_cases_dh_base.json": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json",
        "test_cases_ds_base.json": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_base.json",
        "test_cases_dh_enhanced.json": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_enhanced.json",
        "test_cases_ds_enhanced.json": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_enhanced.json",
        "user_cases.jsonl": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/user_cases.jsonl",
        "attacker_cases_dh.jsonl": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_dh.jsonl",
        "attacker_cases_ds.jsonl": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_ds.jsonl",
    },
    "bipia": {
        "email_test.jsonl": "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/test.jsonl",
        "table_test.jsonl": "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/table/test.jsonl",
        "code_test.jsonl": "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/code/test.jsonl",
        "text_attack_test.json": "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json",
    },
    "xstest": {
        "xstest_v2_prompts.csv": "https://raw.githubusercontent.com/paul-rottger/commonsense-evaluation/main/data/xstest_v2_prompts.csv",
    }
}


def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"📥 Downloading: {os.path.basename(dest_path)} from {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    size_kb = os.path.getsize(dest_path) / 1024
    print(f"   ✓ Saved to {dest_path} ({size_kb:.1f} KB)")


def main():
    print("🚀 Starting download of official benchmark datasets (InjecAgent, BIPIA, XSTest)...")
    for suite, files in URLS.items():
        print(f"\n📂 Suite: {suite.upper()}")
        suite_dir = os.path.join(RAW_DIR, suite)
        for filename, url in files.items():
            dest = os.path.join(suite_dir, filename)
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"   ❌ Failed to download {filename}: {e}")

    print("\n✅ All official raw benchmark datasets downloaded successfully!")


if __name__ == "__main__":
    main()

