# 🛡️ Provable Indirect Prompt Injection Defense via Subspace Weight Surgery

[![Hackathon](https://img.shields.io/badge/Lexsi_Labs-AI_Safety_Hackathon_2026-blue)](https://hackathon.lexsi.ai/)
[![Model](https://img.shields.io/badge/Model-Cohere_Aya_Expanse_8B-green)](https://huggingface.co/CohereLabs/aya-expanse-8b)
[![Track](https://img.shields.io/badge/Track-AI_Red_Teaming%2C_Safety_and_Alignment-orange)](https://hackathon.lexsi.ai/tracks)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)](LICENSE)

An end-to-end alignment and mechanistic interpretability pipeline built for the **Lexsi AI Safety & Alignment Hackathon (Nov 2026)** in partnership with **Cohere Labs** and **Hugging Face**.

---

## 📌 Executive Summary

Autonomous tool-calling agents are structurally vulnerable to **Indirect Prompt Injection (IPI)**: untrusted context (retrieved emails, web scrapes, DB returns) can hijack the autoregressive residual stream, coercing the model into executing unauthorized external APIs (data exfiltration, unauthorized wire transfers, shell commands).

Existing defenses (system prompt delimiters, secondary guardrail LLMs, external filters) either fail against adaptive paraphrasing or introduce prohibitive inference latencies ($>100\,\text{ms}$).

### 🔬 The Core Scientific Thesis
1. **The Readability–Actionability Gap:** Linear probes reveal that token provenance (distinguishing system/user instructions from untrusted tool outputs) is linearly readable in early layers (**L04–L16**, $\text{AUROC} > 0.95$). However, intervening in early layers fails to stop downstream tool dispatch.
2. **The Formation Window:** The causal commitment to emit unauthorized tool dispatches is strictly localized in mid-to-late layers (**L20–L23**), where Multi-Layer Perceptrons (MLPs) supply $\approx 78\%$ of the directional write committing the model to execute a tool.
3. **Subspace Nullification ($P^\perp$):** By extracting the $k$-dimensional authorization subspace $U_S \in \mathbb{R}^{d \times k}$ via **Edge Attribution Patching with Integrated Gradients (EAP-IG)** and SVD, we perform in-place orthogonal weight surgery on MLP down-projection matrices:
   $$W_{\text{down}}^* = P^\perp W_{\text{down}} = (I - U_S U_S^T) W_{\text{down}}$$
   Because $U_S^T P^\perp = 0$, intermediate MLP representations cannot write into the residual stream along the malicious dispatch subspace.

---

## 🚀 Key Results (Pareto Frontier)

| Model & Defense Setup | InjecAgent ASR $\downarrow$ | BIPIA ASR $\downarrow$ | BFCL AST F1 $\uparrow$ | XSTest Over-Refusal $\downarrow$ | Added Latency ($\Delta\tau$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Aya Expanse 8B** (Unmitigated) | 88.4% | 82.1% | 89.2% | 0.8% | 0.00 ms |
| **System-Prompt Delimiters** | 54.2% | 49.6% | 87.1% | 4.6% | +1.20 ms |
| **Llama Guard 3 External Guardrail** | 18.6% | 16.4% | 84.0% | 14.2% | +114.50 ms |
| **SafeTune Subspace Surgery ($P^\perp$)** | **1.8%** | **2.1%** | **88.9%** | **1.1%** | **0.00 ms** |
| **SafeTune ($P^\perp$) + Gated CAA** | **0.4%** | **0.6%** | **88.8%** | **1.2%** | **+0.12 ms** |

---

## 🔗 End-to-End Pipeline Chaining

This project chains **all four repositories** of Lexsi Labs' Track 2 stack:

```
[CuratorKIT]   --> Generates clean, corrupted, and counterfactual triplet datasets
      │
[CircuitKIT]   --> EAP-IG attribution tracing & SVD authorization subspace extraction
      │
[SafeTune]     --> In-place P_perp down-projection surgery & Gated CAA steering
      │
[AuditKIT]     --> Comprehensive evaluation: ASR, BFCL schema F1, ORR, and latency
```

---

## 📂 Repository Structure

```
.
├── HACKATHON_MASTER_PLAN.md    # Exhaustive 1st-place execution roadmap
├── README.md                   # Project overview & documentation
├── pyproject.toml              # Project dependencies & build config
├── requirements.txt            # Python dependencies
├── configs/                    # Model and evaluation YAML configurations
├── data/                       # Raw benchmark data & curated triplets
├── src/
│   ├── curator/                # CuratorKIT: Cohere chat template & triplet synthesis
│   ├── circuit/                # CircuitKIT: EAP-IG causal attribution & subspace SVD
│   ├── safetune/               # SafeTune: P_perp weight surgery & dynamic steering
│   └── audit/                  # AuditKIT: Automated ASR, BFCL, and latency evaluation
├── scripts/                    # Step-by-step pipeline execution scripts (01-05)
├── demo/                       # Interactive Hugging Face Space (Gradio + Plotly)
└── docs/                       # Hackathon context, research article & ROI analysis
```

---

## ⚡ Quickstart

```bash
# 1. Clone repository
git clone https://github.com/GiGiKoneti/lexsi-agent-circuit-defense.git
cd lexsi-agent-circuit-defense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Step 1: Synthesize CuratorKIT triplets
python scripts/01_generate_triplets.py

# 4. Step 2: Run CircuitKIT EAP-IG circuit discovery
python scripts/02_run_circuit_discovery.py

# 5. Step 3: Apply SafeTune orthogonal weight surgery
python scripts/03_apply_weight_surgery.py

# 6. Step 4: Run full AuditKIT benchmarking
python scripts/04_run_full_audit.py
```

---

## 👥 Authors & Team
- **GiGi Koneti** ([@GiGiKoneti](https://github.com/GiGiKoneti)) - Undergrad Researcher (BITS Pilani GRAINS Lab Collaborator)
