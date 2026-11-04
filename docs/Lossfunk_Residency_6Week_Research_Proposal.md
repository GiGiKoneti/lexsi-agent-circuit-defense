# Graph-Geometric Reliability for Vision-Language-Action Policies
## Real-Time Failure Detection via Discrete Forman-Ricci Curvature and Conformal Risk
### 6-Week Research Residency Proposal | House of Lossfunk, Bengaluru

**Author:** GiGi Koneti  
**Affiliation:** Independent Undergrad Researcher | Point Blank Core ML | Research Collaborator with BITS Pilani GRAINS Lab  
**Duration:** 6 Weeks (Full-Time On-Site at the House of Lossfunk)  
**Target Outcome:** Open-source Python/PyTorch library (`frc-vla`) + Workshop Paper Draft targeting NeurIPS / ICLR / CoRL workshops (*Safe & Reliable Robot Learning / Embodied AI*)  
**Compute Budget:** ≤ 18 GPU-Hours total (Single NVIDIA RTX 4090 / A100 or Colab Pro)

---

## 1. Executive Summary & Core Thesis

Generalist Vision-Language-Action (VLA) models such as OpenVLA, Octo, and $\pi_0$ directly map multimodal visual observations and natural language goals into continuous end-effector motor actions. While they generalize well across simple tasks, their reliance on 2D Vision Transformer (ViT) patch tokens creates a critical spatial flaw: **they flatten continuous 3D physical workspaces into unstructured sequence grids**. 

Because standard 2D self-attention operates without differential-geometric priors or kinematic topology, VLA attention maps measure semantic alignment rather than physical interaction stress, spatial bottlenecks, or collision boundaries. Under minor camera shifts, physical perturbations, or visual clutter, VLAs suffer from catastrophic unreliability (crashing into obstacles, missing grasps) while their internal action entropy remains deceptively low and confident.

### The Breakthrough Hypothesis
Physical constraints compress the geometric topology of allowable motion long before physical collision or trajectory divergence occurs. 

By mapping the manipulation scene into a lightweight, dynamic **object-kinematic graph** ($|V| \le 12$), we introduce **Discrete Forman-Ricci Curvature** to monitor the execution path. Unstable or failing VLA trajectories are systematically preceded by localized negative shifts in Forman-Ricci curvature across spatial interaction edges. 

Coupling this zero-parameter geometric metric with a tiny Curvature-Gated GCN (< 35,000 parameters) and **Conformal Risk Control** yields a sub-2 millisecond failure monitor with distribution-free false-warning guarantees, requiring **zero retraining or fine-tuning of multi-billion parameter VLA backbones**.

---

## 2. Theoretical & Mathematical Formulation

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ONLINE PIPELINE OVERVIEW                               │
│                                                                                        │
│  [Visual Observation & State]                                                          │
│              │                                                                         │
│              ▼                                                                         │
│  Entity Keypoint Tracker ────► Dynamic Graph Construction G_t = (V, E_t)               │
│  (Centroids & End-Effector)    Nodes: |V| <= 12 (Positions, Velocities, VLA Attention) │
│                                Edges: Inverse Euclidean Distance Weights               │
│                                              │                                         │
│                                              ▼                                         │
│                                Discrete Forman-Ricci Curvature                         │
│                                F(e) Engine (Zero-parameter, closed form)               │
│                                              │                                         │
│                                              ▼                                         │
│                                2-Layer FRC-GCN Model (< 35k params)                    │
│                                Curvature-Gated Message Passing                         │
│                                              │                                         │
│                                              ▼                                         │
│                                Conformal Risk Boundary (q_1-alpha)                     │
│                                Statistical Coverage Guarantee                          │
│                                              │                                         │
│                                              ▼                                         │
│                     ┌────────────────────────────────────────────────┐                 │
│                     │ Output Verdict: Nominal / Emergency Intervene  │                 │
│                     │ Latency: < 1.8 ms | GPU Compute: Negligible    │                 │
│                     └────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Graph Construction
At each timestep $t$, the workspace is modeled as a dynamic, weighted undirected graph $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$, where nodes $i \in \mathcal{V}$ ($|\mathcal{V}| \le 12$) represent:
* Target object centroids
* Target receptacle points
* Key surface obstacles
* Robot end-effector position

Node features $x_i(t) \in \mathbb{R}^d$ encode 3D spatial coordinates $p_i(t) \in \mathbb{R}^3$, instantaneous velocities $v_i(t) = p_i(t) - p_i(t-1)$, and cross-attention weights extracted from the frozen VLA backbone. Undirected edges $e = (u, v) \in \mathcal{E}_t$ connect nodes within interaction radius $r_{\max}$, weighted by inverse distance $w_e = \|p_u(t) - p_v(t)\|_2^{-1}$.

### 2.2 Discrete Forman-Ricci Curvature Metric
To quantify geometric stress across the interaction graph, discrete Forman-Ricci curvature $\mathbf{F}(e)$ for an edge $e = (u, v)$ with node weights $\gamma_u, \gamma_v$ and edge weight $w_e$ is computed in closed form:

$$\mathbf{F}(e) = w_e \left( \frac{w_e}{\gamma_u} + \frac{w_e}{\gamma_v} \right) - \sum_{e_u \sim e, e_u \neq e} w_{e_u} \frac{\sqrt{w_e w_{e_u}}}{\gamma_u} - \sum_{e_v \sim e, e_v \neq e} w_{e_v} \frac{\sqrt{w_e w_{e_v}}}{\gamma_v}$$

* **Negative shifts ($\mathbf{F}(e) \ll 0$):** Indicate edge contraction under topological bottlenecks, workspace crowding, or impending collision.
* **Positive values ($\mathbf{F}(e) > 0$):** Reflect open, unobstructed motion fields.

### 2.3 Curvature-Gated Graph Convolution (FRC-GCN)
A 2-layer Graph Convolutional Network modulates node message passing with edge curvature:

$$h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \mathbf{F}(i, j) W^{(l)} h_j^{(l)} \right)$$

$$S_{\text{FRC}}(t) = \sigma \left( \text{MLP} \left( \text{Readout}(\{h_i^{(L)}\}_{i \in \mathcal{V}}) \right) \right)$$

where $S_{\text{FRC}}(t) \in [0, 1]$ represents the predicted unreliability score.

### 2.4 Distribution-Free Conformal Risk Control
To prevent false alarms without heuristic threshold guessing, we calibrate a threshold $\hat{q}_{1-\alpha}$ over a held-out calibration set $\mathcal{D}_{\text{cal}}$ of nominal successful trajectories:

$$\hat{q}_{1-\alpha} = \text{Quantile} \left( \left\{ \max_{t \le T} S_{\text{FRC}}(t) \right\}_{m \in \mathcal{D}_{\text{cal}}}, \frac{\lceil (|\mathcal{D}_{\text{cal}}| + 1)(1 - \alpha) \rceil}{|\mathcal{D}_{\text{cal}}|} \right)$$

An intervention flag fires when $S_{\text{FRC}}(t) > \hat{q}_{1-\alpha}$, provably guaranteeing that the false alarm rate under nominal conditions remains bounded by $\alpha \le 0.05$.

---

## 3. Experimental Protocol & Benchmark Suite

### 3.1 Simulation Benchmarks
All experiments run entirely in simulation using:
* **Base Policy:** Frozen **OpenVLA-7B** and **Octo-Small**.
* **Benchmark Suites:** **LIBERO-Spatial** (10 manipulation tasks testing spatial reasoning) and **LIBERO-Goal** (10 manipulation tasks testing goal conditioning).

### 3.2 Controlled Invalidation & Ablation Tests
To prove the geometric metric isolates physical stress rather than general visual shift:
1. **Kinematic Barrier Insertion:** Placing low-contrast obstacles in the path of the gripper.
2. **Adversarial Visual Distractors:** Adding high-contrast background clutter and camera lighting changes without modifying physical obstacles.

**Validation Criterion:** Negative curvature spikes must trigger at least **15 timesteps prior to physical collision** during barrier tests, while remaining within nominal bounds during purely visual shifts.

### 3.3 Target Performance vs. Prior Baselines

| Method | Target Metric | Prior Baseline Performance | Target FRC-GCN Performance |
| :--- | :--- | :--- | :--- |
| **Raw VLA Logit Entropy** | AUROC / ECE | AUROC: 0.51, ECE: 0.34 | **AUROC: ≥ 0.92, ECE: ≤ 0.06** |
| **Sliding Window + ATR** | AUROC / Accuracy | AUROC: 0.78, Acc: 74.2% | **AUROC: ≥ 0.92, Acc: ≥ 88.5%** |
| **Layer-16 Linear Probe (SAFE)** | False Warnings / AUROC | AUROC: 0.81, Warnings: 3.32/ep | **AUROC: ≥ 0.92, Warnings: ≤ 0.15/ep** |
| **Non-Parametric TPC** | AUPRC / Latency | AUPRC: 0.72, Latency: 12.4 ms | **AUPRC: ≥ 0.88, Latency: 1.8 ms** |

---

## 4. 6-Week Detailed Execution Roadmap

```
WEEK 1: Data Pipeline & VLA Hook Extraction
WEEK 2: Vectorized Curvature Engine & Graph Dataset
WEEK 3: Model Architecture & Conformal Risk Calibration
WEEK 4: Closed-Loop Invalidation Testing on LIBERO
WEEK 5: Benchmarking, Baselines & Ablation Studies
WEEK 6: Library Packaging, Interactive Demo & Paper Writing
```

### Week 1: Environment Setup & Trajectory Parsing (Days 1–7)
* Set up LIBERO simulation environment with OpenVLA-7B and Octo-Small inference pipelines.
* Extract 300 nominal and perturbed trajectory rollouts.
* Build `data_extractor.py`: Hook intermediate cross-attention weights and extract 3D entity centroid states.
* **Milestone 1:** Validated data extraction pipeline outputting standardized trajectory logs.

### Week 2: Curvature Engine & Graph Construction (Days 8–14)
* Build `curvature_engine.py`: Vectorized PyTorch implementation of discrete Forman-Ricci edge curvature.
* Verify edge curvature computation takes < 0.5 ms per graph.
* Convert raw trajectory data into PyTorch Geometric `Data` structures with edge curvature attributes.
* **Milestone 2:** Functional, unit-tested discrete differential curvature engine.

### Week 3: Model Architecture & Training (Days 15–21)
* Implement `frc_gcn_model.py`: 2-layer Curvature-Gated GCN (< 35k parameters).
* Train model on trajectory splits using temporally discounted failure loss.
* Implement `conformal_calibrator.py`: Split conformal risk calibration for distribution-free error thresholds ($\alpha = 0.05$).
* **Milestone 3:** Trained FRC-GCN achieving AUROC ≥ 0.90 on offline validation split.

### Week 4: Closed-Loop Evaluation & OOD Invalidation (Days 22–28)
* Integrate FRC-GCN online monitor into the LIBERO evaluation gym loop.
* Execute controlled invalidation tests:
  * Test A: Kinematic barrier insertion (measure lead time before collision).
  * Test B: Adversarial visual distractors (measure false-positive rejection).
* **Milestone 4:** Empirical proof that curvature anomalies trigger ≥ 15 steps prior to physical failure.

### Week 5: Comparative Benchmarking & Ablation (Days 29–35)
* Implement and benchmark all four comparative baselines: Logit Entropy, ATR, Layer Probe, and TPC.
* Run ablation studies:
  * Graph curvature gating vs. standard uniform GCN edge weights.
  * Conformal calibration vs. naive fixed classification thresholds.
  * Number of graph entities ($|\mathcal{V}| = 6, 8, 12$).
* Generate all publication-grade figures, ROC curves, and latency plots.
* **Milestone 5:** Complete experimental results table outperforming all existing baselines.

### Week 6: Packaging, Demo & Paper Writing (Days 36–42)
* Package codebase into clean, pip-installable library: `pip install frc-vla`.
* Build lightweight local dashboard displaying real-time 3D scene graphs with edge curvature color-coding.
* Draft workshop paper:
  * Abstract, Introduction, Differential-Geometric Formulation, Experimental Validation, Related Work.
* **Milestone 6:** Public GitHub repository + complete workshop paper draft ready for submission.

---

## 5. Hardware & Compute Budget Breakdown

Because this project monitors frozen VLA representations via low-dimensional graphs ($|\mathcal{V}| \le 12$) rather than training foundation models, the entire 6-week research program operates comfortably within a **single GPU budget**:

| Phase | Operational Task | Compute Engine | GPU Hours |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Trajectory collection & attention hook extraction | Single RTX 4090 / A100 | 3.5 hrs |
| **Phase 2** | Curvature engine validation & PyG graph construction | CPU / Local GPU | 1.5 hrs |
| **Phase 3** | FRC-GCN model training (50 epochs, AdamW) | Single RTX 4090 / A100 | 2.0 hrs |
| **Phase 4** | Closed-loop LIBERO OOD rollouts & invalidation runs | Single RTX 4090 / A100 | 10.5 hrs |
| **Total** | **Full 6-Week Research Sprint** | **≤ 18 GPU-Hours Total** |

---

## 6. Risks, Fallback Plans, and Self-Sufficiency

| Potential Risk | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- |
| **Dense graph connectivity smooths out edge curvature differences** | Medium | Dynamically sparsify graph using $k$-nearest-neighbor velocity filters to preserve sharp topological bottlenecks. |
| **Simulator entity tracking noise** | Low | Ground-truth coordinates are available natively in LIBERO. For vision-only rollouts, off-the-shelf lightweight point trackers (e.g., CoTracker) provide sub-centimeter stability. |
| **Conformal threshold is overly conservative** | Low | Fall back to standard Youden's Index calibration or relax $\alpha$ from 0.05 to 0.10. |

### Why This Succeeds for a Solo Resident:
* **No external blockers:** Does not rely on physical hardware delivery, custom PCBs, or third-party proprietary APIs.
* **Proven personal strengths:** Direct match with prior research in GNN message passing, PyTorch systems, and mathematical graph modeling.
* **Clear termination criteria:** Week 6 ends with a clean code repository and a complete empirical workshop paper draft.
