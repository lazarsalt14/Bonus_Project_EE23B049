"""
Learning to Reweight Examples (L2RW) - NumPy/sklearn Implementation
Based on: Ren et al., "Learning to Reweight Examples for Robust Deep Learning", ICML 2018

This module implements the L2RW algorithm using a multi-layer perceptron (MLP)
with numpy-based automatic differentiation to reproduce the paper's meta-learning
gradient-based reweighting approach.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  Low-level MLP building blocks (numpy)
# ─────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_one_hot, weights=None):
    """Weighted cross-entropy loss."""
    n = probs.shape[0]
    eps = 1e-12
    log_p = np.log(np.clip(probs, eps, 1.0))
    per_sample = -np.sum(y_one_hot * log_p, axis=1)   # shape (n,)
    if weights is None:
        return per_sample.mean()
    return (weights * per_sample).sum()


class MLP:
    """
    A simple 2-hidden-layer MLP for classification.
    Supports weighted forward/backward passes needed by L2RW.
    """

    def __init__(self, n_in, hidden_sizes, n_out, seed=42):
        rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden_sizes) + [n_out]
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            limit = np.sqrt(2.0 / fan_in)           # He init
            self.W.append(rng.normal(0, limit, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    # ── Forward pass ────────────────────────────────────────
    def forward(self, X):
        """Returns (activations_pre, activations_post, probs)."""
        pre, post = [], []
        h = X
        post.append(h)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pre.append(z)
            if i < len(self.W) - 1:
                h = relu(z)
            else:
                h = softmax(z)
            post.append(h)
        return pre, post, post[-1]

    # ── Gradient of loss w.r.t. parameters ──────────────────
    def grad_params(self, X, y_one_hot, weights=None):
        """
        Standard (or weighted) backprop.
        Returns list of (dW, db) for each layer.
        Also returns per-sample pre-activations and gradients
        (needed by L2RW meta-gradient computation).
        """
        n = X.shape[0]
        pre, post, probs = self.forward(X)

        # Output delta
        delta = probs - y_one_hot                    # (n, n_out)
        if weights is not None:
            delta = delta * weights[:, None]

        grads_W = []
        grads_b = []
        grads_layer = [None] * len(self.W)           # g_l in paper
        grads_layer[-1] = delta / n                  # last layer gradient

        # Backprop through layers
        g = delta
        for i in reversed(range(len(self.W))):
            dW = post[i].T @ g / n
            db = g.mean(axis=0)
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            if i > 0:
                g = g @ self.W[i].T * relu_grad(pre[i - 1])
                grads_layer[i - 1] = g / n

        return list(zip(grads_W, grads_b)), pre, post, grads_layer

    # ── Parameter update (SGD) ───────────────────────────────
    def update(self, grads, lr):
        for i, (dW, db) in enumerate(grads):
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    # ── Snapshot / restore ───────────────────────────────────
    def get_params(self):
        return ([W.copy() for W in self.W],
                [b.copy() for b in self.b])

    def set_params(self, W_list, b_list):
        self.W = [W.copy() for W in W_list]
        self.b = [b.copy() for b in b_list]

    def predict_proba(self, X):
        _, _, probs = self.forward(X)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ─────────────────────────────────────────────────────────────
#  L2RW Meta-weight Computation (Eq. 7, 8, 9 in paper)
# ─────────────────────────────────────────────────────────────

def compute_meta_weights(model, X_train_batch, y_train_batch,
                          X_val, y_val, alpha, n_classes):
    """
    Implements Algorithm 1 from the paper:

    1. Forward noisy batch with eps=0 to get theta_hat
    2. Forward clean validation set through theta_hat
    3. Compute meta-gradient of validation loss w.r.t. eps_i
       using gradient similarity (Eq. 12)
    4. Rectify (max(u, 0)) and normalize

    Returns normalized example weights w_i for each training sample.
    """
    n_train = X_train_batch.shape[0]

    def to_one_hot(y, n):
        oh = np.zeros((len(y), n))
        oh[np.arange(len(y)), y.astype(int)] = 1
        return oh

    y_train_oh = to_one_hot(y_train_batch, n_classes)
    y_val_oh   = to_one_hot(y_val, n_classes)

    # ── Step 1: one gradient step from theta_t (eps=0) ──────
    W_save, b_save = model.get_params()
    grads_train, pre_train, post_train, g_train = model.grad_params(
        X_train_batch, y_train_oh)
    # Compute theta_hat = theta_t - alpha * grad
    W_hat = [W - alpha * dW for W, (dW, _) in zip(model.W, grads_train)]
    b_hat = [b - alpha * db for b, (_, db) in zip(model.b, grads_train)]
    model.set_params(W_hat, b_hat)

    # ── Step 2: validation gradient at theta_hat ────────────
    _, pre_val, post_val, g_val = model.grad_params(X_val, y_val_oh)

    # ── Step 3: meta-gradient (Eq. 12) ──────────────────────
    # u_i = -η * ∂/∂eps_i [val_loss(theta_hat)] |_{eps=0}
    #      ≈ sum over layers l of (z_val^T z_train)(g_val^T g_train)
    # We compute the dot-product similarity for each training example i

    u = np.zeros(n_train)
    n_layers = len(model.W)

    for l in range(n_layers):
        # post[l] is the input activation to layer l+1
        # Shape: (n_train, d_l) and (n_val, d_l)
        z_train = post_train[l]   # (n_train, d_l)
        z_val   = post_val[l]     # (n_val,   d_l)

        g_t = g_train[l]          # (n_train, d_{l+1})
        g_v = g_val[l]            # (n_val,   d_{l+1})

        # For each training example i:
        # u_i += -1/m * sum_j [ (z_val_j . z_train_i)(g_val_j . g_train_i) ]
        act_sim  = z_val   @ z_train.T   # (n_val, n_train)
        grad_sim = g_v     @ g_t.T       # (n_val, n_train)

        # Sum over validation samples j → shape (n_train,)
        contribution = (act_sim * grad_sim).mean(axis=0)
        u -= contribution

    # Restore original parameters
    model.set_params(W_save, b_save)

    # ── Step 4: Rectify and normalize (Eq. 8, 9) ────────────
    w_tilde = np.maximum(u, 0.0)
    total = w_tilde.sum()
    if total == 0:
        w = np.ones(n_train) / n_train
    else:
        w = w_tilde / total

    return w


# ─────────────────────────────────────────────────────────────
#  Full L2RW Classifier (sklearn-compatible)
# ─────────────────────────────────────────────────────────────

class L2RWClassifier(BaseEstimator, ClassifierMixin):
    """
    Learning to Reweight Examples classifier.

    Parameters
    ----------
    hidden_sizes : tuple
        Hidden layer sizes for the MLP.
    lr : float
        Learning rate (alpha in the paper).
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for training examples.
    val_batch_size : int
        Mini-batch size for validation (m in paper, can equal |D_val|).
    seed : int
        Random seed.
    """

    def __init__(self, hidden_sizes=(64, 32), lr=0.01, n_epochs=200,
                 batch_size=64, val_batch_size=32, seed=42):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.seed = seed

    def fit(self, X_train, y_train, X_val, y_val, verbose=False):
        """
        Train using the L2RW algorithm.

        Parameters
        ----------
        X_train, y_train : noisy/imbalanced training data
        X_val, y_val     : small, clean validation set
        """
        self.classes_ = np.unique(np.concatenate([y_train, y_val]))
        n_classes = len(self.classes_)
        self.n_classes_ = n_classes

        # Re-map labels to 0..K-1
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_tr = np.array([label_map[yi] for yi in y_train])
        y_v  = np.array([label_map[yi] for yi in y_val])

        self.scaler_ = StandardScaler()
        X_tr = self.scaler_.fit_transform(X_train)
        X_v  = self.scaler_.transform(X_val)

        rng = np.random.default_rng(self.seed)
        n_in = X_tr.shape[1]
        self.model_ = MLP(n_in, self.hidden_sizes, n_classes, seed=self.seed)

        self.train_losses_ = []
        self.val_losses_   = []
        self.weight_history_ = []

        n = len(y_tr)

        def to_one_hot(y, k):
            oh = np.zeros((len(y), k))
            oh[np.arange(len(y)), y.astype(int)] = 1
            return oh

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n)
            epoch_weights = []

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                X_b = X_tr[batch_idx]
                y_b = y_tr[batch_idx]

                # Sample validation mini-batch
                val_idx = rng.choice(len(y_v),
                                     size=min(self.val_batch_size, len(y_v)),
                                     replace=False)
                X_vb = X_v[val_idx]
                y_vb = y_v[val_idx]

                # Compute meta-weights
                w = compute_meta_weights(
                    self.model_, X_b, y_b, X_vb, y_vb,
                    self.lr, n_classes)
                epoch_weights.extend(w.tolist())

                # Update with weighted loss
                y_b_oh = to_one_hot(y_b, n_classes)
                grads, _, _, _ = self.model_.grad_params(X_b, y_b_oh, weights=w)
                self.model_.update(grads, self.lr)

            # Record losses
            _, _, tr_probs = self.model_.forward(X_tr)
            y_tr_oh = to_one_hot(y_tr, n_classes)
            tr_loss = cross_entropy_loss(tr_probs, y_tr_oh)

            _, _, v_probs = self.model_.forward(X_v)
            y_v_oh = to_one_hot(y_v, n_classes)
            v_loss = cross_entropy_loss(v_probs, y_v_oh)

            self.train_losses_.append(tr_loss)
            self.val_losses_.append(v_loss)
            self.weight_history_.append(epoch_weights[:self.batch_size])

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}: train_loss={tr_loss:.4f}, "
                      f"val_loss={v_loss:.4f}")

        return self

    def predict_proba(self, X):
        X_s = self.scaler_.transform(X)
        return self.model_.predict_proba(X_s)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
