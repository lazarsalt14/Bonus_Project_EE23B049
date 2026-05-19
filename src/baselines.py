"""
Baseline classifiers for comparison with L2RW.

Includes three ML classifiers taught in course:
1. Logistic Regression (with class weighting option)
2. Random Forest (with class weighting option)
3. Support Vector Machine (SVM) with RBF kernel

Also includes simple reweighting strategies from the paper:
- PROPORTION  : inverse-frequency weighting
- RESAMPLE    : class-balanced resampling
- HARD_MINING : up-weight high-loss examples (for imbalance)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


# ─────────────────────────────────────────────────────────────
#  Course Baselines
# ─────────────────────────────────────────────────────────────

def get_logistic_regression(class_weight=None, seed=42):
    """Logistic Regression (L2 regularised)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            
            random_state=seed
        ))
    ])


def get_random_forest(class_weight=None, seed=42):
    """Random Forest classifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1
        ))
    ])


def get_svm(class_weight=None, seed=42):
    """SVM with RBF kernel and probability estimates."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight=class_weight,
            probability=True,
            random_state=seed,
            gamma="scale"
        ))
    ])


# ─────────────────────────────────────────────────────────────
#  Paper Reweighting Strategies (for MLP baseline)
# ─────────────────────────────────────────────────────────────

def proportion_weights(y):
    """
    PROPORTION: weight each example by inverse class frequency.
    wi = 1 / freq(class_i), normalized to sum to 1.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts / len(y)))
    raw = np.array([1.0 / freq[yi] for yi in y])
    return raw / raw.sum()


def resample_balanced(X, y, seed=42):
    """
    RESAMPLE: class-balanced resampling to majority class size.
    Returns (X_resampled, y_resampled).
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [], []
    rng = np.random.default_rng(seed)
    for c in classes:
        mask = y == c
        Xc, yc = X[mask], y[mask]
        if len(yc) < max_count:
            idx = rng.choice(len(yc), size=max_count - len(yc), replace=True)
            Xc = np.vstack([Xc, Xc[idx]])
            yc = np.concatenate([yc, yc[idx]])
        X_parts.append(Xc)
        y_parts.append(yc)
    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def hard_mining_weights(losses, top_fraction=0.5):
    """
    HARD MINING: assign weight 1 to top-loss examples, 0 to rest.
    top_fraction: fraction of examples to keep.
    """
    threshold = np.percentile(losses, 100 * (1 - top_fraction))
    w = (losses >= threshold).astype(float)
    total = w.sum()
    if total == 0:
        return np.ones(len(losses)) / len(losses)
    return w / total


# ─────────────────────────────────────────────────────────────
#  Vanilla MLP (no reweighting) – for ablation
# ─────────────────────────────────────────────────────────────

class VanillaMLP:
    """
    Vanilla MLP trained with standard unweighted SGD.
    Used as the BASELINE in Table 1/2 of the paper.
    """

    def __init__(self, hidden_sizes=(64, 32), lr=0.01,
                 n_epochs=200, batch_size=64, seed=42):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y, verbose=False):
        from l2rw import MLP, cross_entropy_loss, relu
        from sklearn.preprocessing import StandardScaler

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[yi] for yi in y])

        self.scaler_ = StandardScaler()
        X_s = self.scaler_.fit_transform(X)

        n_in = X_s.shape[1]
        self.model_ = MLP(n_in, self.hidden_sizes, n_classes, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        n = len(y_mapped)
        self.train_losses_ = []

        def to_one_hot(y, k):
            oh = np.zeros((len(y), k))
            oh[np.arange(len(y)), y.astype(int)] = 1
            return oh

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb = X_s[batch]
                yb = y_mapped[batch]
                yb_oh = to_one_hot(yb, n_classes)
                grads, _, _, _ = self.model_.grad_params(Xb, yb_oh)
                self.model_.update(grads, self.lr)

            _, _, probs = self.model_.forward(X_s)
            loss = cross_entropy_loss(probs, to_one_hot(y_mapped, n_classes))
            self.train_losses_.append(loss)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        X_s = self.scaler_.transform(X)
        return self.model_.predict_proba(X_s)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
