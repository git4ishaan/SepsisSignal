# SepsisSignal — Model Efficiency & Deployment Feasibility Benchmark Report

**Project:** SepsisSignal Research Prototype  
**Scope:** Quantitative Efficiency, Latency, Memory, and Deployment Feasibility Benchmark  
**Hardware & Environment:** Intel Core (16 Physical / 22 Logical Cores), 31.61 GB RAM, Windows 11, Python 3.13.7, PyTorch 2.14.1+cpu  
**Evaluation Constraint:** Zero modifications to live serving paths, database schemas, socket protocol, or frozen model artifacts.  

> [!IMPORTANT]
> **Safety Disclaimer:** Research prototype, not for clinical decision-making.

---

## 1. Executive Summary & Key Benchmark Metrics

We benchmarked all candidate models under identical hardware and input conditions using high-resolution monotonic timing (`time.perf_counter_ns`), separate warm-up cycles (20 warm-up runs discarded), and repeated trials (100 repetitions per configuration):
1. **Logistic Regression** (`models/model.joblib`) — current live serving model
2. **XGBoost Classifier** (`models/xgboost_model.joblib`) — offline tree comparison
3. **PyTorch GRU** (`models/lstm_model.pt`) — offline causal recurrent sequence model
4. **Soft-Voting Ensemble** (`SepsisEnsemble`) — equal-weight probability average

### Headline Efficiency Comparison

| Candidate Model | Isolated Latency at 48h (Median p50) | Isolated Latency at 48h (95th %ile) | Isolated Latency at 48h (99th %ile) | 48-Hour Stay Total Streaming Time | Batch Throughput (pred / sec) | Peak Traced Heap Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Live)** | **0.42 ms** | **0.51 ms** | **0.65 ms** | **33.57 ms** | **5,426,525** | **26.88 KB** |
| **XGBoost (Offline)** | **0.56 ms** | **0.67 ms** | **0.97 ms** | **39.89 ms** | **1,687,308** | **255.60 KB** |
| **PyTorch GRU (Offline)** | **5.94 ms** | **7.99 ms** | **9.47 ms** | **159.16 ms** | **14,526** | **34.13 KB** |
| **Soft-Voting Ensemble** | **8.27 ms** | **9.09 ms** | **9.62 ms** | **251.78 ms** | **14,364** | **284.10 KB** |

---

## 2. Isolated Single-Point-in-Time Prediction Latency

Isolated point-in-time prediction measures the complete pipeline cost: observation ingestion, causal window filtering, 22-dimensional feature extraction, normalization, and model forward pass.

### Latency vs Patient History Length (1 to 48 Hours)

| Model | 1 Hour (p50) | 6 Hours (p50) | 12 Hours (p50) | 24 Hours (p50) | 36 Hours (p50) | 48 Hours (p50) | Complexity Scaling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.3203 ms | 0.3591 ms | 0.4764 ms | 0.4647 ms | 0.4008 ms | **0.4214 ms** | **$O(1)$ constant** |
| **XGBoost** | 0.5086 ms | 0.4735 ms | 0.5459 ms | 0.5200 ms | 0.5267 ms | **0.5582 ms** | **$O(1)$ constant** |
| **PyTorch GRU** | 0.3242 ms | 0.7326 ms | 1.0354 ms | 2.4253 ms | 3.0831 ms | **5.9432 ms** | **$O(W^2)$ scaling** |
| **Soft-Voting Ensemble** | 1.7283 ms | 2.4087 ms | 2.2227 ms | 3.6806 ms | 4.8878 ms | **8.2656 ms** | **$O(W^2)$ dominated by GRU** |

*Key Findings:*
1. **Tabular Models (LR and XGBoost):** Exhibit strictly constant $O(1)$ latency regardless of patient history length (~0.42 ms for LR, ~0.56 ms for XGBoost). They extract only the target hour's 22 features.
2. **PyTorch GRU:** Latency grows from 0.32 ms at $H=1$ to 5.94 ms at $H=48$ (**18.3x latency increase**). This growth is driven by the sequence window unrolling and per-step feature extraction.
3. **Soft-Voting Ensemble:** Sits at ~8.27 ms at $H=48$, **~19.6x slower than Logistic Regression** and **~14.8x slower than XGBoost**.

---

## 3. Sequential Streaming Simulation (1 to 48 Hours)

In bedside clinical monitoring, observations arrive sequentially hour by hour. We simulated streaming ingestion of 48-hour patient stays:
- **Logistic Regression:** Cumulative stay processing time = **33.57 ms** (average **0.70 ms / hour**).
- **XGBoost:** Cumulative stay processing time = **39.89 ms** (average **0.83 ms / hour**).
- **PyTorch GRU:** Cumulative stay processing time = **159.16 ms** (average **3.32 ms / hour**).
- **Soft-Voting Ensemble:** Cumulative stay processing time = **251.78 ms** (average **5.25 ms / hour**).

---

## 4. Throughput & Memory Profiling

### 4.1 Batch Inference Throughput
- **Logistic Regression:** **5,426,525 predictions / second**
- **XGBoost:** **1,687,308 predictions / second**
- **PyTorch GRU:** **14,526 predictions / second**
- **Soft-Voting Ensemble:** **14,364 predictions / second**

*Throughput Ratio:* Logistic Regression processes **~378x more predictions per second** than the ensemble, and XGBoost processes **~117x more predictions per second** than the ensemble.

### 4.2 Heap Memory Allocation & Process RSS
- **Logistic Regression:** Peak heap allocation during inference = **26.88 KB**. Extremely lightweight; negligible garbage collection pressure.
- **XGBoost:** Peak heap allocation = **255.60 KB** (due to internal decision-tree histogram structures).
- **PyTorch GRU:** Peak heap allocation = **34.13 KB**.
- **Soft-Voting Ensemble:** Peak heap allocation = **284.10 KB**. Process RSS remained stable at ~516 MB (with zero memory leaks over 10,000 iterations).

---

## 5. Computational Complexity Audit & Optimization Analysis

### 5.1 Audit of Existing Implementation Overhead
We audited `src/ensemble.py` (`predict_point_in_time`) to identify unnecessary historical recomputations:
1. **$O(W^2)$ Feature Re-extraction Overhead:**
   - In `predict_point_in_time`, when evaluating target hour $T$, the GRU requires a sequence window of up to 48 hours.
   - The current implementation calls:
     ```python
     for r in window:
         h = int(r["hour"])
         v_step, _ = extract_features_point_in_time(sorted_causal, target_hour=h, imputer=self.imputer)
         seq_feats.append(v_step.flatten())
     ```
   - For an isolated hour-48 prediction, this calls `extract_features_point_in_time` **48 separate times**. Across a 48-hour stay, it performs $1 + 2 + \dots + 48 = 1,176$ feature vector extractions instead of 48.
2. **Full Recurrent Forward Pass on Each Hour:**
   - At each hour $t$, the full sequence tensor $(1, t, 22)$ is re-fed through the GRU from $h_0 = 0$, recomputing hidden states $h_1, \dots, h_{t-1}$ that were already calculated at prior hours.

### 5.2 Identified Production Optimization Opportunities (Not implemented in live path)
If the ensemble or GRU were to be considered for production deployment, the following optimizations would eliminate ~85% of latency:
1. **Feature Vector Caching:** Cache each hour's 22-dimensional feature vector in SQLite/memory when received. Generating the 48-step sequence would then require 1 vector extraction + 47 memory array lookups ($O(1)$ instead of $O(W)$).
2. **Recurrent Hidden State Caching:** Maintain the GRU hidden state vector $h_{t-1} \in \mathbb{R}^{48}$ in the patient session. When observation $x_t$ arrives, execute a single `GRUCell(x_t, h_{t-1})` forward step ($O(1)$ forward pass) rather than re-running the full unrolled sequence of 48 timesteps.

---

## 6. Comparison at Validation-Selected Operating Points

To assess the clinical trade-off between efficiency and predictive utility, all models were evaluated on the **233,127 held-out test hours** (6,048 patients, 4,006 positive hours, 1.72% prevalence) at their pre-selected validation operating points:

| Model & Operating Point | Threshold ($\tau$) | Selection Criterion | Test Recall | Test Precision | Test Specificity | Test Alert Rate (/100h) | Test $F_1$ | Median Latency (48h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Live)** | 0.4200 | Targeted Recall $\ge 70\%$ | 73.74% | 3.01% | 58.49% | 42.06 | 0.0579 | **0.42 ms** |
| **XGBoost (Offline)** | 0.3900 | Targeted Recall $\ge 70\%$ | 76.44% | 3.80% | 66.13% | 34.60 | 0.0723 | **0.56 ms** |
| **PyTorch GRU (Offline)** | 0.4200 | Targeted Recall $\ge 70\%$ | 76.66% | 3.88% | 66.77% | 33.98 | 0.0738 | **5.94 ms** |
| **Ensemble (Uncalibrated)** | 0.4110 | Targeted Recall $\ge 70\%$ | 75.29% | 4.15% | 69.59% | 31.19 | 0.0786 | **8.27 ms** |
| **Ensemble (Calibrated Screening)** | 0.0131 | Targeted Recall $\ge 70\%$ | 75.41% | 4.15% | 69.52% | 31.26 | 0.0786 | **8.27 ms** |
| **Ensemble (Calibrated Max-$F_1$)** | 0.0526 | Unconstrained Max $F_1$ | 34.32% | 11.20% | 95.24% | 5.27 | 0.1688 | **8.27 ms** |

---

## 7. Determinism & Causal Invariance Verification

- **Determinism Check:** Executing 100 repeated inferences on identical observations produced identical probabilities with **zero numerical variance** ($0.0$ difference across all decimal places).
- **Causal Invariance Check:** Appending future observations with severe shock vitals (Hour 25: HR=175, SBP=50, WBC=32, Temp=40.5°C) produced **bit-for-bit identical outputs** for Hour 24 across all models, verifying zero future information leakage.

---

## 8. Technical Recommendation

### Recommendation: **Option B — Keep XGBoost as the Primary Offline Candidate and Retain Logistic Regression for Live Bedside Serving.**

#### Detailed Technical Justification:
1. **Live Production Path (Logistic Regression):**
   - **Recommendation:** Retain Logistic Regression on the current live inference socket.
   - **Why:** LR executes in **0.42 ms** with **26 KB** memory overhead and reaches **5.4M predictions/second**. It achieves 73.7% test recall with zero external dependencies (no PyTorch, no GPU). For bedside edge gateways with strict hardware limits, this remains the most reliable baseline.
2. **Offline Analytics & Server-Side Batch Screening (XGBoost vs Ensemble):**
   - **Statistical Parity:** In our 1,000 paired cluster bootstrap replicates, the ensemble's AUPRC difference over XGBoost was $\Delta = -0.0009$ with a 95% CI of `[-0.0113, +0.0084]` (containing zero). The ensemble **did not establish statistically significant superiority over standalone XGBoost on AUPRC**.
   - **Efficiency Dominance of XGBoost:**
     - Latency: XGBoost is **~14.8x faster** than the ensemble (0.56 ms vs 8.27 ms).
     - Throughput: XGBoost delivers **1,687,308 pred/sec** vs **14,364 pred/sec** for the ensemble (**117x higher throughput**).
     - Architecture Simplicity: XGBoost is a standalone tabular model requiring zero sequence buffers, zero PyTorch runtime, and zero hidden state tracking.
   - **Conclusion:** Deploying the three-model ensemble adds substantial architectural complexity (synchronizing tabular feature pipelines with recurrent sequence tensors) for no statistically verifiable AUPRC gain over standalone XGBoost. XGBoost represents the superior Pareto-optimal candidate for high-throughput batch and server-side sepsis risk scoring.

---

## 9. Benchmark Artifacts & Plots

Generated under `models/results/ensemble_audit/benchmark/`:
- **Machine-Readable Benchmark Results:** [benchmark_results.json](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/benchmark_results.json)
- **Latency vs History Length Plot:** [latency_by_history_length.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/latency_by_history_length.png)
- **Latency Percentiles at 48h:** [latency_percentiles_comparison.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/latency_percentiles_comparison.png)
- **Cumulative 48-Hour Streaming Latency:** [streaming_cumulative_time.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/streaming_cumulative_time.png)
- **Throughput vs Peak Memory:** [throughput_memory_comparison.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/throughput_memory_comparison.png)
