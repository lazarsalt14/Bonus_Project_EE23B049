"""
Experiments for Learning to Reweight Examples (L2RW) replication.

Experiment 1: Class Imbalance (Credit dataset)
  - Vary imbalance ratio from 2:1 to 20:1
  - Compare L2RW vs Logistic Regression, Random Forest, SVM,
    Vanilla MLP, PROPORTION weighting, RESAMPLE

Experiment 2: Noisy Labels (Phishing dataset)
  - Vary noise rate from 0% to 50%
  - Noise types: uniform, background
  - Compare L2RW vs same baselines

Metrics: Accuracy, F1 (macro), AUC-ROC, G-mean
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

from datasets import (load_credit_dataset, load_phishing_dataset,
                       print_dataset_summary)
from baselines import (get_logistic_regression, get_random_forest,
                        get_svm, VanillaMLP, proportion_weights,
                        resample_balanced)
from l2rw import L2RWClassifier


# ─────────────────────────────────────────────────────────────
#  Evaluation Metrics
# ─────────────────────────────────────────────────────────────

def geometric_mean(y_true, y_pred):
    """G-mean: geometric mean of per-class recalls."""
    cm = confusion_matrix(y_true, y_pred)
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return float(np.prod(recalls) ** (1 / len(recalls)))


def compute_auc(model, X_test, y_test):
    """Multi-class AUC (OvR macro)."""
    classes = np.unique(y_test)
    try:
        proba = model.predict_proba(X_test)
        if len(classes) == 2:
            return roc_auc_score(y_test, proba[:, 1])
        else:
            y_bin = label_binarize(y_test, classes=classes)
            return roc_auc_score(y_bin, proba, multi_class="ovr",
                                  average="macro")
    except Exception:
        return np.nan


def evaluate(model, X_test, y_test, name=""):
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred, average="macro", zero_division=0)
    gm   = geometric_mean(y_test, pred)
    auc  = compute_auc(model, X_test, y_test)
    return {"model": name, "accuracy": acc, "f1_macro": f1,
            "gmean": gm, "auc": auc}


# ─────────────────────────────────────────────────────────────
#  Experiment 1: Class Imbalance
# ─────────────────────────────────────────────────────────────

def run_imbalance_experiment(imbalance_ratios=None, seed=42):
    if imbalance_ratios is None:
        imbalance_ratios = [2, 5, 10, 15, 20]

    all_results = {}

    for ratio in imbalance_ratios:
        print(f"\n{'='*60}")
        print(f"  IMBALANCE RATIO  {ratio}:1")
        print(f"{'='*60}")

        data = load_credit_dataset(n_samples=2000,
                                    imbalance_ratio=ratio, seed=seed)
        print_dataset_summary(data)

        X_tr, y_tr = data["X_train"], data["y_train"]
        X_v,  y_v  = data["X_val"],   data["y_val"]
        X_te, y_te = data["X_test"],  data["y_test"]

        results = []

        # 1. Logistic Regression (no weighting)
        print("  Fitting Logistic Regression...")
        lr = get_logistic_regression(seed=seed)
        lr.fit(X_tr, y_tr)
        results.append(evaluate(lr, X_te, y_te, "Logistic Reg"))

        # 2. Logistic Regression (balanced)
        lr_bal = get_logistic_regression(class_weight="balanced", seed=seed)
        lr_bal.fit(X_tr, y_tr)
        results.append(evaluate(lr_bal, X_te, y_te, "Logistic Reg (balanced)"))

        # 3. Random Forest
        print("  Fitting Random Forest...")
        rf = get_random_forest(seed=seed)
        rf.fit(X_tr, y_tr)
        results.append(evaluate(rf, X_te, y_te, "Random Forest"))

        # 4. Random Forest (balanced)
        rf_bal = get_random_forest(class_weight="balanced", seed=seed)
        rf_bal.fit(X_tr, y_tr)
        results.append(evaluate(rf_bal, X_te, y_te, "Random Forest (balanced)"))

        # 5. SVM
        print("  Fitting SVM...")
        svm = get_svm(seed=seed)
        svm.fit(X_tr, y_tr)
        results.append(evaluate(svm, X_te, y_te, "SVM"))

        # 6. SVM (balanced)
        svm_bal = get_svm(class_weight="balanced", seed=seed)
        svm_bal.fit(X_tr, y_tr)
        results.append(evaluate(svm_bal, X_te, y_te, "SVM (balanced)"))

        # 7. Vanilla MLP (baseline)
        print("  Fitting Vanilla MLP...")
        vanilla = VanillaMLP(hidden_sizes=(64, 32), lr=0.01,
                              n_epochs=150, batch_size=64, seed=seed)
        vanilla.fit(X_tr, y_tr)
        results.append(evaluate(vanilla, X_te, y_te, "Vanilla MLP"))

        # 8. PROPORTION weighting (MLP)
        print("  Fitting MLP + PROPORTION...")
        from l2rw import MLP, cross_entropy_loss
        from sklearn.preprocessing import StandardScaler
        w_prop = proportion_weights(y_tr)
        prop_mlp = VanillaMLP(hidden_sizes=(64, 32), lr=0.01,
                               n_epochs=150, batch_size=64, seed=seed)
        # Inject weights at fit time — we use a lightweight wrapper
        prop_mlp_fitted = _fit_weighted_mlp(X_tr, y_tr, w_prop, seed=seed)
        results.append(evaluate(prop_mlp_fitted, X_te, y_te,
                                 "MLP + PROPORTION"))

        # 9. RESAMPLE
        print("  Fitting MLP + RESAMPLE...")
        X_res, y_res = resample_balanced(X_tr, y_tr, seed=seed)
        resample_mlp = VanillaMLP(hidden_sizes=(64, 32), lr=0.01,
                                   n_epochs=150, batch_size=64, seed=seed)
        resample_mlp.fit(X_res, y_res)
        results.append(evaluate(resample_mlp, X_te, y_te, "MLP + RESAMPLE"))

        # 10. L2RW (ours)
        print("  Fitting L2RW...")
        l2rw = L2RWClassifier(hidden_sizes=(64, 32), lr=0.01,
                               n_epochs=150, batch_size=64,
                               val_batch_size=len(y_v), seed=seed)
        l2rw.fit(X_tr, y_tr, X_v, y_v, verbose=False)
        results.append(evaluate(l2rw, X_te, y_te, "L2RW (ours)"))

        for r in results:
            print(f"    {r['model']:30s}  "
                  f"Acc={r['accuracy']:.3f}  "
                  f"F1={r['f1_macro']:.3f}  "
                  f"AUC={r['auc']:.3f}  "
                  f"GMean={r['gmean']:.3f}")

        all_results[ratio] = results

    return all_results


# ─────────────────────────────────────────────────────────────
#  Experiment 2: Noisy Labels
# ─────────────────────────────────────────────────────────────

def run_noise_experiment(noise_rates=None, noise_type="uniform", seed=42):
    if noise_rates is None:
        noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    all_results = {}

    for nr in noise_rates:
        print(f"\n{'='*60}")
        print(f"  NOISE TYPE={noise_type}  RATE={nr:.0%}")
        print(f"{'='*60}")

        data = load_phishing_dataset(n_samples=2000, noise_rate=nr,
                                      noise_type=noise_type, seed=seed)
        print_dataset_summary(data)

        X_tr, y_tr = data["X_train"], data["y_train"]
        X_v,  y_v  = data["X_val"],   data["y_val"]
        X_te, y_te = data["X_test"],  data["y_test"]

        results = []

        # Baselines
        for name, model in [
            ("Logistic Reg",   get_logistic_regression(seed=seed)),
            ("Random Forest",  get_random_forest(seed=seed)),
            ("SVM",            get_svm(seed=seed)),
        ]:
            print(f"  Fitting {name}...")
            model.fit(X_tr, y_tr)
            results.append(evaluate(model, X_te, y_te, name))

        # Vanilla MLP
        print("  Fitting Vanilla MLP...")
        vanilla = VanillaMLP(hidden_sizes=(64, 32), lr=0.01,
                              n_epochs=150, batch_size=64, seed=seed)
        vanilla.fit(X_tr, y_tr)
        results.append(evaluate(vanilla, X_te, y_te, "Vanilla MLP"))

        # L2RW
        print("  Fitting L2RW...")
        l2rw = L2RWClassifier(hidden_sizes=(64, 32), lr=0.01,
                               n_epochs=150, batch_size=64,
                               val_batch_size=len(y_v), seed=seed)
        l2rw.fit(X_tr, y_tr, X_v, y_v, verbose=False)
        results.append(evaluate(l2rw, X_te, y_te, "L2RW (ours)"))

        for r in results:
            print(f"    {r['model']:30s}  "
                  f"Acc={r['accuracy']:.3f}  "
                  f"F1={r['f1_macro']:.3f}  "
                  f"AUC={r['auc']:.3f}")

        all_results[nr] = results

    return all_results


# ─────────────────────────────────────────────────────────────
#  Helper: fit a pre-weighted MLP
# ─────────────────────────────────────────────────────────────

def _fit_weighted_mlp(X, y, weights, hidden=(64, 32), lr=0.01,
                      n_epochs=150, batch=64, seed=42):
    """Train a VanillaMLP with fixed pre-computed sample weights."""
    from l2rw import MLP, cross_entropy_loss
    from sklearn.preprocessing import StandardScaler

    class WeightedMLP:
        pass

    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y_m = np.array([label_map[yi] for yi in y])
    n_classes = len(classes)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = MLP(Xs.shape[1], hidden, n_classes, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(y_m)

    def to_one_hot(yy, k):
        oh = np.zeros((len(yy), k)); oh[np.arange(len(yy)), yy.astype(int)] = 1
        return oh

    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            Xb, yb, wb = Xs[bi], y_m[bi], weights[bi]
            wb = wb / (wb.sum() + 1e-12) * len(wb)
            grads, _, _, _ = model.grad_params(Xb, to_one_hot(yb, n_classes),
                                                weights=wb)
            model.update(grads, lr)

    obj = WeightedMLP()
    obj.classes_ = classes
    obj.scaler_ = scaler
    obj.model_ = model
    obj.predict_proba = lambda Xt: model.predict_proba(scaler.transform(Xt))
    obj.predict = lambda Xt: classes[model.predict_proba(
        scaler.transform(Xt)).argmax(axis=1)]
    return obj


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("\n" + "="*60)
    print("  EXPERIMENT 1: CLASS IMBALANCE")
    print("="*60)
    imb_results = run_imbalance_experiment(
        imbalance_ratios=[2, 5, 10, 15, 20], seed=42)

    print("\n" + "="*60)
    print("  EXPERIMENT 2: NOISY LABELS (UNIFORM)")
    print("="*60)
    noise_results_uniform = run_noise_experiment(
        noise_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        noise_type="uniform", seed=42)

    print("\n" + "="*60)
    print("  EXPERIMENT 2: NOISY LABELS (BACKGROUND)")
    print("="*60)
    noise_results_bg = run_noise_experiment(
        noise_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        noise_type="background", seed=42)

    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(type(obj))

    all_out = {
        "imbalance": {str(k): v for k, v in imb_results.items()},
        "noise_uniform": {str(k): v for k, v in noise_results_uniform.items()},
        "noise_background": {str(k): v for k, v in noise_results_bg.items()},
    }
    os.makedirs("../results", exist_ok=True)
    with open("../results/all_results.json", "w") as f:
        json.dump(all_out, f, indent=2, default=convert)
    print("\nResults saved to results/all_results.json")
