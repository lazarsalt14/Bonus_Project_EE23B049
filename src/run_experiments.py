"""
Full experiment runner — uses tuned hyperparameters.
Runs both experiments and saves all results to JSON.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from datasets import load_credit_dataset, load_phishing_dataset, print_dataset_summary
from baselines import (get_logistic_regression, get_random_forest, get_svm,
                        VanillaMLP, proportion_weights, resample_balanced)
from l2rw import L2RWClassifier

# ── Hyper-params ────────────────────────────────────────────
L2RW_PARAMS   = dict(hidden_sizes=(64, 32), lr=0.25, n_epochs=200,
                      batch_size=64, val_batch_size=40, seed=42)
VANILLA_PARAMS = dict(hidden_sizes=(64, 32), lr=0.25, n_epochs=200,
                       batch_size=64, seed=42)


def geometric_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r  = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return float(np.prod(r) ** (1 / len(r)))


def compute_auc(model, X, y):
    classes = np.unique(y)
    try:
        p = model.predict_proba(X)
        if len(classes) == 2:
            return float(roc_auc_score(y, p[:, 1]))
        yb = label_binarize(y, classes=classes)
        return float(roc_auc_score(yb, p, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def evaluate(model, X, y, name):
    pred = model.predict(X)
    return {
        "model":    name,
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "gmean":    float(geometric_mean(y, pred)),
        "auc":      compute_auc(model, X, y),
    }


def fit_and_eval_baselines(X_tr, y_tr, X_v, y_v, X_te, y_te, seed=42):
    results = []

    def run(name, model, Xtr=X_tr, ytr=y_tr):
        print(f"    {name}...", flush=True)
        model.fit(Xtr, ytr)
        results.append(evaluate(model, X_te, y_te, name))

    run("Logistic Reg",              get_logistic_regression(seed=seed))
    run("Logistic Reg (balanced)",   get_logistic_regression("balanced", seed))
    run("Random Forest",             get_random_forest(seed=seed))
    run("Random Forest (balanced)",  get_random_forest("balanced", seed))
    run("SVM",                       get_svm(seed=seed))
    run("SVM (balanced)",            get_svm("balanced", seed))

    print("    Vanilla MLP...", flush=True)
    v = VanillaMLP(**VANILLA_PARAMS)
    v.fit(X_tr, y_tr)
    results.append(evaluate(v, X_te, y_te, "Vanilla MLP"))

    # PROPORTION
    print("    MLP + PROPORTION...", flush=True)
    wp = proportion_weights(y_tr)
    vp = _weighted_mlp(X_tr, y_tr, wp, seed)
    results.append(evaluate(vp, X_te, y_te, "MLP + PROPORTION"))

    # RESAMPLE
    print("    MLP + RESAMPLE...", flush=True)
    Xr, yr = resample_balanced(X_tr, y_tr, seed)
    vr = VanillaMLP(**VANILLA_PARAMS); vr.fit(Xr, yr)
    results.append(evaluate(vr, X_te, y_te, "MLP + RESAMPLE"))

    # L2RW
    print("    L2RW...", flush=True)
    params = dict(L2RW_PARAMS)
    params["val_batch_size"] = len(y_v)
    m = L2RWClassifier(**params)
    m.fit(X_tr, y_tr, X_v, y_v)
    results.append(evaluate(m, X_te, y_te, "L2RW (ours)"))

    for r in results:
        print(f"      {r['model']:30s}  "
              f"Acc={r['accuracy']:.3f}  F1={r['f1_macro']:.3f}  "
              f"AUC={r['auc']:.3f}  GMean={r['gmean']:.3f}")
    return results


def _weighted_mlp(X, y, weights, seed=42):
    from l2rw import MLP
    from sklearn.preprocessing import StandardScaler
    classes = np.unique(y)
    lm = {c: i for i, c in enumerate(classes)}
    ym = np.array([lm[yi] for yi in y])
    nc = len(classes)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = MLP(Xs.shape[1], (64, 32), nc, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(ym)

    def oh(yy):
        o = np.zeros((len(yy), nc))
        o[np.arange(len(yy)), yy.astype(int)] = 1
        return o

    for _ in range(200):
        idx = rng.permutation(n)
        for s in range(0, n, 64):
            bi = idx[s:s+64]
            Xb, yb, wb = Xs[bi], ym[bi], weights[bi]
            wb = wb / (wb.sum() + 1e-12) * len(wb)
            g, _, _, _ = model.grad_params(Xb, oh(yb), weights=wb)
            model.update(g, 0.25)

    class M:
        def predict_proba(self, Xt):
            return model.predict_proba(sc.transform(Xt))
        def predict(self, Xt):
            return classes[self.predict_proba(Xt).argmax(axis=1)]

    return M()


# ── Experiment 1 ─────────────────────────────────────────────

def exp_imbalance(ratios=(2, 5, 10, 15, 20), seed=42):
    all_res = {}
    for ratio in ratios:
        print(f"\n{'='*55}\n  IMBALANCE {ratio}:1\n{'='*55}")
        d = load_credit_dataset(n_samples=2000, imbalance_ratio=ratio, seed=seed)
        print_dataset_summary(d)
        all_res[ratio] = fit_and_eval_baselines(
            d["X_train"], d["y_train"],
            d["X_val"],   d["y_val"],
            d["X_test"],  d["y_test"], seed)
    return all_res


# ── Experiment 2 ─────────────────────────────────────────────

def exp_noise(rates=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
              noise_type="uniform", seed=42):
    all_res = {}
    for nr in rates:
        print(f"\n{'='*55}\n  NOISE {noise_type} {nr:.0%}\n{'='*55}")
        d = load_phishing_dataset(n_samples=2000, noise_rate=nr,
                                   noise_type=noise_type, seed=seed)
        print_dataset_summary(d)
        all_res[nr] = fit_and_eval_baselines(
            d["X_train"], d["y_train"],
            d["X_val"],   d["y_val"],
            d["X_test"],  d["y_test"], seed)
    return all_res


if __name__ == "__main__":
    def cvt(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o

    os.makedirs("../results", exist_ok=True)

    print("\n╔══════════════════════════════════════════╗")
    print("║   L2RW EXPERIMENTS — full run             ║")
    print("╚══════════════════════════════════════════╝")

    imb = exp_imbalance((2, 5, 10, 15, 20))
    nu  = exp_noise((0.0, 0.1, 0.2, 0.3, 0.4, 0.5), "uniform")
    nb  = exp_noise((0.0, 0.1, 0.2, 0.3, 0.4, 0.5), "background")

    out = {
        "imbalance":         {str(k): v for k, v in imb.items()},
        "noise_uniform":     {str(k): v for k, v in nu.items()},
        "noise_background":  {str(k): v for k, v in nb.items()},
    }
    path = "../results/all_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=cvt)
    print(f"\nAll results saved to {path}")
