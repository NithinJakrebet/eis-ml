import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)  # Avoid division by zero
    mu = np.where(np.isnan(mu), 0.0, mu)   # Fully-NaN column -> mean 0
    # Impute missing cells to the column mean (0 in standardized space) so GPR,
    # which rejects NaN, works on datasets with irregular EIS coverage (e.g. sparseEIS).
    Xs = np.nan_to_num((X - mu) / sig, nan=0.0)
    return Xs, mu, sig


def _standardize_apply(X, mu, sig):
    return np.nan_to_num((X - mu) / sig, nan=0.0)


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


def train_capacity_gpr(X_train, y_train, gpr_params=None, subset_size=None):
    """Zhang-faithful capacity GPR predictor.

    Mirrors Multi_T_EIS_Capacity_GPR.m: an *isotropic* squared-exponential
    covariance (covSEiso, a single shared length scale) with a *learned*
    Gaussian noise term (likGauss, sn~=0.1). ARD is intentionally NOT used
    here (see train_ard_diagnostic) because Zhang used ARD only as a
    feature-importance diagnostic, not as the regressor.

    subset_size: if set, hyperparameters are optimized on a stratified subset
    of this many points, then the kernel is frozen and the posterior is
    conditioned on the FULL training set (optimizer disabled). Exact-GP
    hyperparameter optimization is O(n^3) per likelihood eval, so optimizing
    on a few hundred points and conditioning on all of them is far cheaper
    while keeping full-data predictions. With the 3-hyperparameter isotropic
    kernel this subset optimization is robust (unlike 140-dim ARD on a subset).
    If None, hyperparameters are optimized directly on the full data.
    """
    if gpr_params is None:
        gpr_params = {}

    alpha = gpr_params.get('alpha', 1e-10)
    normalize_y = gpr_params.get('normalize_y', True)
    n_restarts = gpr_params.get('n_restarts_optimizer', 5)
    random_state = gpr_params.get('random_state', 42)

    Xs, mu, sig = _standardize_fit(X_train)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )

    if subset_size is not None and subset_size < len(y_train):
        # Stage 1: learn hyperparameters on a stratified subset
        Xsub, ysub = _stratified_subsample(Xs, y_train, n=subset_size, seed=random_state)
        gpr_opt = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts, random_state=random_state,
        ).fit(Xsub, ysub)
        # Stage 2: freeze hyperparameters, condition on all data
        model = GaussianProcessRegressor(
            kernel=gpr_opt.kernel_, alpha=alpha, normalize_y=normalize_y,
            optimizer=None, random_state=random_state,
        ).fit(Xs, y_train)
    else:
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts, random_state=random_state,
        ).fit(Xs, y_train)

    return dict(model=model, mu=mu, sig=sig, kind="capacity_iso", cols=None)


def train_ard_diagnostic(X_train, y_train, gpr_params=None, subset_size=300):
    """Separate ARD-SE model used ONLY to extract per-frequency relevance weights.

    Mirrors Zhang's ARD_GPR.m (covSEard): one length scale per feature, so
    exp(-length_scale) ranks feature importance. This is a diagnostic, not the
    predictor. Optimized on a stratified subset for speed (default 300), since
    ARD has one hyperparameter per feature and we only need the weights.
    """
    if gpr_params is None:
        gpr_params = {}

    Xs, mu, sig = _standardize_fit(X_train)
    if subset_size is not None and subset_size < len(y_train):
        Xs, y_train = _stratified_subsample(
            Xs, y_train, n=subset_size, seed=gpr_params.get('random_state', 42)
        )

    n_features = Xs.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gpr_params.get('alpha', 1e-10),
        normalize_y=gpr_params.get('normalize_y', True),
        n_restarts_optimizer=gpr_params.get('n_restarts_optimizer', 1),
        random_state=gpr_params.get('random_state', 42),
    ).fit(Xs, y_train)

    return dict(model=model, mu=mu, sig=sig, kind="capacity", cols=None)


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


# Alias: works for both the isotropic predictor and the ARD diagnostic bundles.
predict = predict_fast

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
