import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)  # Avoid division by zero
    return (X - mu) / sig, mu, sig


def _standardize_apply(X, mu, sig):
    return (X - mu) / sig


def _stratified_subsample(X, y, n=300, seed=42):
    rng = np.random.RandomState(seed)
    
    qs = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    yb = np.digitize(y, qs)
    
    idx = np.arange(len(y))
    sel = []
    for b in np.unique(yb):
        bi = idx[yb == b]
        if bi.size == 0:
            continue
        rng.shuffle(bi)
        k = max(1, int(round(n * bi.size / len(y))))
        sel.extend(bi[:k])
    
    sel = np.array(sel[:n], int)
    return X[sel], y[sel]

def train_capacity_gpr_fast(
    X_train, 
    y_train, 
    subset_size=300, 
    top_k_freqs=None,
    kernel_params=None,
    gpr_params=None
):
    """
    Fast two-stage GPR training
    
    Args:
        X_train, y_train: Training data
        subset_size: Samples for hyperparameter learning
        top_k_freqs: If set, use ARD to select top-K features
        kernel_params: Dict with kernel configuration
        gpr_params: Dict with GPR base parameters
    """
    
    Xs_all, mu, sig = _standardize_fit(X_train)
    Xs_sub, y_sub = _stratified_subsample(Xs_all, y_train, n=subset_size)
    
    # Build kernel from config or use defaults
    n_features = X_train.shape[1]
    ard_init = np.ones(n_features)
    
    if kernel_params is None:
        kernel_params = {}
    
    kernel = (
        ConstantKernel(
            kernel_params.get('constant_value', 1.0),
            tuple(kernel_params.get('constant_bounds', [1e-2, 1e2]))
        ) * 
        RBF(
            length_scale=ard_init,
            length_scale_bounds=tuple(kernel_params.get('rbf_length_scale_bounds', [1e-2, 1e2]))
        ) +
        WhiteKernel(
            noise_level=kernel_params.get('white_noise_level', 1e-3),
            noise_level_bounds=kernel_params.get('white_noise_bounds', 'fixed')
        )
    )
    
    # Build GPR parameters from config or use defaults
    if gpr_params is None:
        gpr_params = {}
    
    gpr_sub = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gpr_params.get('alpha', 0.0),
        normalize_y=gpr_params.get('normalize_y', True),
        n_restarts_optimizer=gpr_params.get('n_restarts_optimizer', 1),
        random_state=gpr_params.get('random_state', 42)
    ).fit(Xs_sub, y_sub)
    
    cols = np.arange(n_features)
    if top_k_freqs is not None:
        w = ard_frequency_weights(dict(
            model=gpr_sub, mu=mu, sig=sig, kind="capacity"
        ))
        n_freq = n_features // 2
        w_mean = (w[:n_freq] + w[n_freq:]) / 2.0
        top = np.argsort(w_mean)[::-1][:top_k_freqs]
        cols = np.sort(np.concatenate([top, top + n_freq]))
        Xs_all = Xs_all[:, cols]
    
    # Stage 2: Fit full data with frozen hyperparameters
    frozen_kernel = gpr_sub.kernel_
    gpr_full = GaussianProcessRegressor(
        kernel=frozen_kernel,
        alpha=gpr_params.get('alpha', 0.0),
        normalize_y=gpr_params.get('normalize_y', True),
        optimizer=None,  # No optimization - use learned hyperparameters
        random_state=gpr_params.get('random_state', 42)
    ).fit(Xs_all, y_train)
    
    return dict(
        model=gpr_full,
        mu=mu,
        sig=sig,
        kind="capacity",
        cols=cols
    )


def predict_fast(bundle, X_test):
    """
    Predict with trained GPR model
    
    Returns:
        mean: Predicted capacity values
        std: Uncertainty estimates (standard deviation)
    """
    Xs = _standardize_apply(X_test, bundle["mu"], bundle["sig"])
    
    # Apply feature selection if used during training
    if "cols" in bundle and bundle["cols"] is not None:
        Xs = Xs[:, bundle["cols"]]
    
    mean, std = bundle["model"].predict(Xs, return_std=True)
    return mean, std

def ard_frequency_weights(bundle):
    """
    Extract ARD importance weights from trained model
    
    Returns exp(-length_scale) for each feature:
    - High weight = short length scale = important feature
    - Low weight = long length scale = unimportant feature
    """
    if bundle.get("kind") != "capacity":
        raise ValueError("ARD weights only available for capacity models")
    
    # Navigate kernel tree to find RBF component
    k = bundle["model"].kernel_
    rbf = None
    
    # Check common kernel structures
    if hasattr(k, "k1") and hasattr(k.k1, "k2") and isinstance(k.k1.k2, RBF):
        rbf = k.k1.k2
    elif hasattr(k, "k1") and isinstance(k.k1, RBF):
        rbf = k.k1
    elif hasattr(k, "k2") and isinstance(k.k2, RBF):
        rbf = k.k2
    
    if rbf is None or not hasattr(rbf, "length_scale"):
        raise RuntimeError("Could not find RBF kernel with length_scale")
    
    ls = np.atleast_1d(rbf.length_scale)
    return np.exp(-ls)  # Convert length scale to importance weight
