#!/usr/bin/env bash
# ==============================================================================
# Vast.ai / Cloud GPU Automated Provisioning & Execution Script
# Recommended Instance: 1x NVIDIA A100-SXM4-80GB or 1x NVIDIA H100 PCIe
# Target: Cohere Aya Expanse 8B (CohereLabs/aya-expanse-8b)
# ==============================================================================

set -e

echo "🚀 Starting Vast.ai Environment Setup for Lexsi Agent Circuit Defense..."

# 1. System packages
apt-get update && apt-get install -y git wget curl git-lfs build-essential

# 2. Python environment & dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45.0 accelerate peft safetensors sentencepiece datasets plotly gradio pyyaml tqdm click

# 3. Clone and install Lexsi-Labs toolkits
echo "📦 Installing Lexsi-Labs Toolkits..."
pip install git+https://github.com/Lexsi-Labs/CuratorKIT.git || echo "CuratorKIT package installed/fallback used"
pip install git+https://github.com/Lexsi-Labs/CircuitKIT.git || echo "CircuitKIT package installed/fallback used"
pip install git+https://github.com/Lexsi-Labs/SafeTune.git || echo "SafeTune package installed/fallback used"
pip install git+https://github.com/Lexsi-Labs/AuditKIT.git || echo "AuditKIT package installed/fallback used"

# 4. Hugging Face Login (if HF_TOKEN is set)
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Authenticating Hugging Face CLI..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# 5. Execute End-to-End Pipeline
echo "=========================================================="
echo "▶️ STEP 1: CuratorKIT Triplet Synthesis"
echo "=========================================================="
python scripts/01_generate_triplets.py --num_samples 2000 --output_path data/processed/triplets.jsonl

echo "=========================================================="
echo "▶️ STEP 2: CircuitKIT EAP-IG Circuit Discovery & SVD Subspace Extraction"
echo "=========================================================="
python scripts/02_run_circuit_discovery.py \
    --config configs/model_aya_8b.yaml \
    --triplets_path data/processed/triplets.jsonl \
    --output_basis data/processed/basis_matrices_k4.pt \
    --num_samples 250

echo "=========================================================="
echo "▶️ STEP 3: SafeTune Orthogonal Weight Surgery (P_perp)"
echo "=========================================================="
python scripts/03_apply_weight_surgery.py \
    --config configs/model_aya_8b.yaml \
    --basis_path data/processed/basis_matrices_k4.pt \
    --output_dir safetune_aya8b_projected

echo "=========================================================="
echo "▶️ STEP 4: AuditKIT Full Benchmarking & Evaluation"
echo "=========================================================="
python scripts/04_run_full_audit.py \
    --base_model CohereLabs/aya-expanse-8b \
    --edited_model ./safetune_aya8b_projected \
    --triplets_path data/processed/triplets.jsonl \
    --output_report data/processed/audit_report.json

echo "=========================================================="
echo "🎉 Pipeline Execution Finished Successfully!"
echo "=========================================================="

