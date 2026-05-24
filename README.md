# Learning to Reweight Examples (L2RW) — Replication Study

> Bonus Assignment | Course: DA5400 | Roll No: EE23B049
Replication of **"Learning to Reweight Examples for Robust Deep Learning"** (Ren et al., ICML 2018) on two datasets, with comparison against three course baselines.

**Paper**: [arXiv:1803.09050](https://arxiv.org/abs/1803.09050) 

**Video**: https://youtu.be/xU3EYTdYbQw 

**Report**: `report/da5400_report_ee23b049.pdf` 

---

## Project Structure

```
l2rw/
├── src/
│   ├── l2rw.py              # Core L2RW algorithm (NumPy MLP + meta-weight computation)
│   ├── baselines.py         # Logistic Reg, Random Forest, SVM, Vanilla MLP
│   ├── datasets.py          # Dataset loading and preprocessing
│   ├── run_experiments.py   # Full experiment runner (Exp 1 + Exp 2)
│   └── visualize.py         # Figure generation
├── results/
│   └── all_results.json     # Saved experiment results
├── figures/                 # All generated figures
├── report/
│   ├── report.tex           
│   └── report_bonus_assignment.pdf  
└── README.md
```

---

## Experiments

### Experiment 1: Class Imbalance (Credit Dataset)
- Varies imbalance ratio from **2:1 to 20:1**
- Metrics: F1 Macro, AUC-ROC, G-mean, Accuracy

### Experiment 2: Noisy Labels (Phishing Dataset)
- Noise rates: **0% to 50%**
- Noise types: **uniform flip** and **background flip**
- Metrics: Accuracy, F1 Macro, AUC-ROC

---

## Setup

```bash
# Python 3.8+
pip install numpy scikit-learn matplotlib pandas seaborn

# Run all experiments (~5-10 min on CPU)
cd src
python3 run_experiments.py

# Generate figures
python3 visualize.py
```

---

## Key Results

| Method | F1 @10:1 imbalance | Accuracy @40% noise |
|---|---|---|
| Logistic Reg | 0.697 | 0.718 |
| Random Forest | 0.700 | 0.785 |
| SVM | 0.788 | 0.807 |
| Vanilla MLP | 0.860 | 0.550 |
| **L2RW (ours)** | **0.827** | **0.245** |

L2RW confirms the paper's core claim: example weights correctly identify and suppress noisy labels. On tabular data, ensemble methods remain competitive.

---

## Algorithm Overview

L2RW solves a bi-level optimisation problem at each training step:

1. **Forward** noisy batch with ε=0 weights
2. **Backward** on training loss → θ̂
3. **Forward** clean validation set through θ̂
4. **Backward on backward** → compute ∂val_loss/∂εᵢ (gradient similarity)
5. **Rectify + normalize** → example weights wᵢ
6. **Reweighted update** → θₜ₊₁

See `src/l2rw.py` → `compute_meta_weights()` for the implementation.

---

## References

- Ren et al. (2018). *Learning to Reweight Examples for Robust Deep Learning*. ICML 2018.
- GitHub code by authors: https://github.com/uber-research/learning-to-reweight-examples
