# gpr_scripts_slim.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel

# ---- config
GPR_CAPACITY_BASE = dict(alpha=0.0, normalize_y=True, n_restarts_optimizer=5, random_state=42)
GPR_RUL_BASE      = dict(alpha=0.0, normalize_y=True, n_restarts_optimizer=5, random_state=42)

# ---- utils
def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return (X - mu) / sig, mu, sig

def _standardize_apply(X, mu, sig):
    return (X - mu) / sig

def _capacity_kernel(n_features):
    ls = np.ones(n_features)
    return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=ls, length_scale_bounds=(1e-3, 1e3)) \
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

def _rul_kernel():
    return ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=0.0) \
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

# ---- main API
def train_capacity_gpr(X_train, y_train, params=None):
    Xs, mu, sig = _standardize_fit(X_train)
    cfg = dict(GPR_CAPACITY_BASE);  cfg.update(params or {})
    cfg["kernel"] = _capacity_kernel(X_train.shape[1])
    model = GaussianProcessRegressor(**cfg).fit(Xs, y_train)
    return dict(model=model, mu=mu, sig=sig, kind="capacity")

def train_rul_gpr(X_train, y_train, params=None):
    Xs, mu, sig = _standardize_fit(X_train)
    cfg = dict(GPR_RUL_BASE);  cfg.update(params or {})
    cfg["kernel"] = _rul_kernel()
    model = GaussianProcessRegressor(**cfg).fit(Xs, y_train)
    return dict(model=model, mu=mu, sig=sig, kind="rul")

def train_gpr(X_train, y_train, kind="capacity", params=None):
    return train_capacity_gpr(X_train, y_train, params) if kind=="capacity" \
           else train_rul_gpr(X_train, y_train, params)

def predict(bundle, X_test):
    Xs = _standardize_apply(X_test, bundle["mu"], bundle["sig"])
    if "cols" in bundle and bundle["cols"] is not None:
        Xs = Xs[:, bundle["cols"]]
    mean, std = bundle["model"].predict(Xs, return_std=True)
    return mean, std

def ard_frequency_weights(bundle):
    if bundle.get("kind") != "capacity":
        raise ValueError("ARD weights available only for capacity model.")
    k = bundle["model"].kernel_
    rbf = None
    if hasattr(k, "k1") and hasattr(k.k1, "k2") and isinstance(k.k1.k2, RBF): rbf = k.k1.k2
    elif hasattr(k, "k1") and isinstance(k.k1, RBF): rbf = k.k1
    elif hasattr(k, "k2") and isinstance(k.k2, RBF): rbf = k.k2
    else:
        for node in (getattr(k, "k1", None), getattr(k, "k2", None)):
            for sub in (node, getattr(node, "k1", None), getattr(node, "k2", None)):
                if isinstance(sub, RBF): rbf = sub; break
            if rbf is not None: break
    if rbf is None or not hasattr(rbf, "length_scale"):
        raise RuntimeError("RBF with length_scale not found.")
    ls = np.atleast_1d(rbf.length_scale)
    return np.exp(-ls)

# ---- fast two-stage trainer (+ optional ARD pruning)
def _stratified_subsample(X, y, n=300, n_bins=10, seed=42):
    rng = np.random.RandomState(seed)
    qs = np.quantile(y, np.linspace(0,1,n_bins+1)[1:-1])
    yb = np.digitize(y, qs)
    idx = np.arange(len(y))
    sel = []
    for b in np.unique(yb):
        bi = idx[yb==b]
        if bi.size == 0: continue
        rng.shuffle(bi)
        k = max(1, int(round(n * bi.size / len(y))))
        sel.extend(bi[:k])
    sel = np.array(sel[:n], int)
    return X[sel], y[sel]

def train_capacity_gpr_fast(X_train, y_train, params=None, subset_size=300, top_k_freqs=None):
    Xs_all, mu, sig = _standardize_fit(X_train)
    Xs_sub, y_sub = _stratified_subsample(Xs_all, y_train, n=subset_size)
    n_features = X_train.shape[1]

    ard_init = np.ones(n_features)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=ard_init, length_scale_bounds=(1e-2, 1e2)) \
           + WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")

    cfg = dict(GPR_CAPACITY_BASE); cfg.update(params or {})
    cfg.update(dict(kernel=kernel, n_restarts_optimizer=1, random_state=42))
    gpr_sub = GaussianProcessRegressor(**cfg).fit(Xs_sub, y_sub)

    cols = np.arange(n_features)
    if top_k_freqs is not None:
        w = ard_frequency_weights(dict(model=gpr_sub, mu=mu, sig=sig, kind="capacity"))
        n_freq = n_features // 2
        w_mean = (w[:n_freq] + w[n_freq:]) / 2.0
        top = np.argsort(w_mean)[::-1][:top_k_freqs]
        cols = np.sort(np.concatenate([top, top + n_freq]))
        Xs_all = Xs_all[:, cols]

    frozen = gpr_sub.kernel_
    gpr_full = GaussianProcessRegressor(kernel=frozen, alpha=0.0, normalize_y=True,
                                        optimizer=None, random_state=42).fit(Xs_all, y_train)
    return dict(model=gpr_full, mu=mu, sig=sig, kind="capacity", cols=cols)

def predict_fast(bundle, X_test):
    return predict(bundle, X_test)
