"""
Dataset loading and preprocessing for L2RW experiments.

Two UCI datasets are used:
1. Credit Card Fraud Detection (class imbalance)
   - Highly imbalanced: ~0.17% fraud cases
   - Source: UCI / synthetic version via sklearn

2. Phishing Websites (noisy labels experiment)
   - Binary classification
   - We inject synthetic label noise to study robustness

Both datasets are preprocessed to support:
- Variable imbalance ratios (Section 4.1 analogue)
- Uniform and background label noise (Section 4.2 analogue)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_credit_dataset(n_samples=2000, imbalance_ratio=10, seed=42):
    """
    Credit-card-style imbalanced binary dataset.
    Uses sklearn's make_classification with controlled imbalance.

    Parameters
    ----------
    n_samples      : total number of samples
    imbalance_ratio: majority:minority ratio (e.g. 10 = 10:1)
    seed           : random seed

    Returns
    -------
    dict with keys: X_train, y_train, X_test, y_test,
                    X_val, y_val, feature_names
    """
    n_minority = n_samples // (imbalance_ratio + 1)
    n_majority = n_samples - n_minority
    weights = [n_majority / n_samples, n_minority / n_samples]

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=0.0,
        random_state=seed
    )

    # Train/val/test split: 70/10/20
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.125, stratify=y_tr, random_state=seed)

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,   y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        name=f"Credit (ratio={imbalance_ratio}:1)"
    )


def load_phishing_dataset(n_samples=2000, noise_rate=0.3,
                           noise_type="uniform", seed=42):
    """
    Phishing-website-style multi-class dataset with synthetic label noise.

    Parameters
    ----------
    n_samples   : total samples
    noise_rate  : fraction of labels to corrupt
    noise_type  : 'uniform' or 'background' (analogous to paper Section 4.2)
    seed        : random seed

    Returns
    -------
    dict with keys: X_train, y_train (noisy), X_test, y_test (clean),
                    X_val, y_val (clean), y_train_clean, noise_mask
    """
    rng = np.random.default_rng(seed)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=seed
    )
    classes = np.unique(y)
    n_classes = len(classes)

    # Train/val/test split
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train_clean, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=seed)

    # Inject noise into training labels
    n_train = len(y_train_clean)
    y_train_noisy = y_train_clean.copy()
    n_noisy = int(noise_rate * n_train)
    noise_idx = rng.choice(n_train, size=n_noisy, replace=False)
    noise_mask = np.zeros(n_train, dtype=bool)
    noise_mask[noise_idx] = True

    if noise_type == "uniform":
        # Each noisy label flips to a random different class
        for i in noise_idx:
            other = [c for c in classes if c != y_train_clean[i]]
            y_train_noisy[i] = rng.choice(other)
    elif noise_type == "background":
        # All noisy labels flip to class 0 (background)
        y_train_noisy[noise_idx] = 0
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return dict(
        X_train=X_train, y_train=y_train_noisy,
        y_train_clean=y_train_clean, noise_mask=noise_mask,
        X_val=X_val,    y_val=y_val,
        X_test=X_test,  y_test=y_test,
        feature_names=[f"feature_{i}" for i in range(X.shape[1])],
        name=f"Phishing ({noise_type} noise, rate={noise_rate})"
    )


def get_imbalance_stats(y):
    """Return class distribution statistics."""
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}


def print_dataset_summary(data):
    """Print a tidy summary of a dataset dict."""
    print(f"Dataset : {data['name']}")
    print(f"  Train : {len(data['y_train'])} samples, "
          f"dist={get_imbalance_stats(data['y_train'])}")
    if 'y_train_clean' in data:
        n_noisy = data['noise_mask'].sum()
        print(f"  Noise  : {n_noisy} labels corrupted "
              f"({100*n_noisy/len(data['y_train']):.1f}%)")
    print(f"  Val   : {len(data['y_val'])} samples, "
          f"dist={get_imbalance_stats(data['y_val'])}")
    print(f"  Test  : {len(data['y_test'])} samples, "
          f"dist={get_imbalance_stats(data['y_test'])}")
    print()
