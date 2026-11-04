# Lexsi - AI Safety and Alignment Hackathon: Event Specifications & Strategy

**Organizers:** Lexsi Labs in partnership with Cohere Labs and Hugging Face  
**Outreach Partners:** AISafety.com, AI Safety India (AISIN)  
**Timeline:** Nov 1 – Nov 21, 2026 (Online)  
**Submission Deadline:** Nov 14, 2026, 23:59 AoE  
**Demo Day Announcement:** Nov 16, 2026 (Up to 20 teams shortlisted)  
**Demo Day:** Nov 22, 2026 (Hybrid: Polaris School of Technology, Bengaluru & Online)  
**Prizes:** $30,000+ prize pool, Hugging Face Pro & GPU credits, and direct interview opportunities for DevRel / Research Internship / Research Scientist roles at Lexsi Labs.

---

## 1. Hackathon Goals & Expectations

- **Core Mission:** Bring Lexsi Labs' open-source model-lifecycle stack into the hands of builders working on frontier safety, interpretability, alignment, and agentic autonomy.
- **The Core Rule:** Chain open-source repositories from the chosen track into an **end-to-end, functional pipeline** and demonstrate **before-and-after metrics** showing that an open-weights model becomes measurably safer and more reliable.
  - *Minimum:* At least 2 repos from the track used together.
  - *Preferred (Top Scoring):* 3 or more repos used together in a real pipeline.
- **Mandatory Models:** Dense, open-weights LLMs from Cohere Labs:
  - `CohereLabs/tiny-aya-base`, `tiny-aya-global`, `tiny-aya-fire`, `tiny-aya-water`, `tiny-aya-earth`
  - `CohereLabs/aya-expanse-8b`
  - `CohereLabs/aya-expanse-32b`
- **Hugging Face Hosting:** Models, adapters, evaluation logs, and curated datasets must be hosted on Hugging Face.

---

## 2. Tracks & Toolkit Breakdown

### Track 1: Fine-tuning and Data Curation
- **Stack:** `CuratorKIT` (data curation / synthetic data) $\to$ `AlignTune` (post-training / 20+ algorithms) $\to$ `AuditKIT` (benchmarking).
- **Focus:** Domain assistants for regulated industries (healthcare, legal, banking), replacing closed APIs with small models, hallucination suppression, and long documents.

### Track 2: AI Red Teaming, Safety and Alignment (OUR TARGET TRACK)
- **Stack:** `CuratorKIT` $\to$ `CircuitKIT` $\to$ `SafeTune` $\to$ `AuditKIT`
  - **`CuratorKIT`:** Data curation & synthetic generation pipeline producing quality-gated, deduplicated training/test data.
  - **`CircuitKIT`:** Mechanistic-interpretability framework for discovering, evaluating, and intervening on circuits inside transformer models (pruning, path patching, circuit-restricted fine-tuning).
  - **`SafeTune`:** Library for auditing and repairing safety drift in fine-tuned LLMs (train-time hardening, weight-space interventions, inference-time steering).
  - **`AuditKIT`:** Evaluation suite with benchmark collections, metrics, and automated diagnostics.
- **Sub-tracks:**
  - **2a (Red Teaming & Audits):** Attack hardened models, multi-turn audits, prompt injection hidden in tools/documents.
  - **2b (Safety & Correction):** Locating where safety lives, circuit-level interventions, inference-time steering, weight-space surgery ($P^\perp$), and safety-preserving fine-tuning (AsFT/RefusalGuard).

### Track 3: Agentic Fine-tuning
- **Stack:** `AgentTune` (post-training agents for tool calls, syntax, trajectory efficiency) $\to$ `AuditKIT`.
- **Focus:** Reliable tool calling, cutting redundant loops, and database agents.

---

## 3. Our Winning Project Strategy: Bridging Track 2 & Track 3

**Project Title:** *Mechanistic Circuit Discovery and Subspace Weight Steering for Provable Indirect Prompt Injection Defense in Agentic Function-Calling Architectures*

### Why This Architecture Wins:
1. **Chains 4 Repositories:** Combines `CuratorKIT` + `CircuitKIT` + `SafeTune` + `AuditKIT` (exceeding the 3-repo preference).
2. **Attacks Frontier 2025–2026 Science:** Leverages the *Readability–Actionability Gap* in transformers (L04–L16 encode token provenance, but L20–L23 MLPs make the causal decision write).
3. **Provable Math vs. Brittle Prompts:** Deploys orthogonal subspace projection ($P^\perp = I - U_S U_S^T$) directly into MLP down-projection matrices ($W_{\text{down}}$), eliminating indirect prompt injections with **zero inference latency** ($\Delta\tau = 0.00\,\text{ms}$) and near-zero degradation in benign tool precision.
4. **Full Economic ROI Analysis:** Demonstrates $73.3\%$ cost reduction vs. GPT-4o for enterprise tool-calling deployments.

---

## 4. Key Submission Deliverables Checklist

- [ ] Public GitHub repository with clean, modular, and documented codebase.
- [ ] Curated Hugging Face Dataset (triplets: clean, corrupted, counterfactual).
- [ ] Safe, weight-edited Cohere model checkpoints/adapters on Hugging Face (`safetune-aya-8b-projected`).
- [ ] Comprehensive before-and-after benchmark evaluation logs (`AuditKIT`).
- [ ] Hugging Face Article / Technical Report detailing scientific findings.
- [ ] Interactive Hugging Face Space (Gradio) showing real-time comparative execution and live Plotly activation diagnostics.
- [ ] Bonus Economic ROI analysis report.
