# gpr_ard.py
# Gaussian Process Regression with ARD (per-feature length scales), sklearn implementation.
# Mirrors the structure of your XGBoost ensemble helpers.

import os
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

import config

@dataclass
class GPRModelBundle:
    scaler: StandardScaler
    gp: GaussianProcessRegressor

def _build_ard_kernel(
    n_features: int,
    noise_level: float = 1e-6,
    length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
    noise_level_bounds: Tuple[float, float] = (1e-12, 1e-3)
) -> RBF:
    rbf = RBF(length_scale=np.ones(n_features), length_scale_bounds=length_scale_bounds)
    white = WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
    return rbf + white

def _extract_rbf_from_kernel(kernel):
    """Return the RBF part from a (RBF + WhiteKernel) sum, regardless of ordering."""
    k1, k2 = getattr(kernel, "k1", None), getattr(kernel, "k2", None)
    if isinstance(k1, RBF):
        return k1
    if isinstance(k2, RBF):
        return k2
    raise RuntimeError("Fitted kernel does not contain an RBF component.")

# =========================
# Public training / predict
# =========================
def train_gpr_model(X_train: np.ndarray,
                    y_train: np.ndarray,
                    model_params: Optional[Dict] = None) -> GPRModelBundle:
    if model_params is None and hasattr(config, "GPR_PARAMS"):
        model_params = config.GPR_PARAMS.copy()
    elif model_params is None:
        model_params = {}

    noise_level = model_params.get("noise_level", 1e-6)
    length_scale_bounds = model_params.get("length_scale_bounds", (1e-5, 1e5))
    noise_level_bounds = model_params.get("noise_level_bounds", (1e-12, 1e-3))
    normalize_y = model_params.get("normalize_y", True)
    n_restarts_optimizer = model_params.get("n_restarts_optimizer", 3)
    random_state = model_params.get("random_state", 42)

    # 1) scale X (common best practice for GP stability)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train)

    # 2) ARD kernel (vector length_scale of size d)
    d = Xs.shape[1]
    kernel = _build_ard_kernel(
        n_features=d,
        noise_level=noise_level,
        length_scale_bounds=length_scale_bounds,
        noise_level_bounds=noise_level_bounds,
    )

    # 3) GP regressor
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,  # noise handled by WhiteKernel
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state
    )

    # 4) fit
    gp.fit(Xs, y_train)

    return GPRModelBundle(scaler=scaler, gp=gp)

def predict_gpr(model_bundle: GPRModelBundle,
                X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict mean and std for X_test using the trained GPR model.
    Returns:
        y_mean: (n_samples,)
        y_std:  (n_samples,)
    """
    Xs = model_bundle.scaler.transform(X_test)
    y_mean, y_std = model_bundle.gp.predict(Xs, return_std=True)
    return y_mean, y_std

# =========================
# Save / load
# =========================
def save_gpr_model(model_bundle: GPRModelBundle,
                   filename: str = "gpr_ard.pkl") -> str:
    """
    Save the entire bundle (scaler + gp) in one file under config.MODELS_DIR.
    Returns absolute path to saved file.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, filename)
    joblib.dump(model_bundle, path)
    print(f"Saved GPR model to {path}")
    return path

def load_gpr_model(filename: str = "gpr_ard.pkl") -> GPRModelBundle:
    """
    Load a previously saved bundle.
    """
    path = os.path.join(config.MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model_bundle: GPRModelBundle = joblib.load(path)
    print(f"Loaded GPR model from {path}")
    return model_bundle

# =========================
# ARD importance utilities
# =========================
def get_ard_importance(model_bundle: GPRModelBundle,
                       f_grid: Optional[np.ndarray] = None,
                       n_freqs: Optional[int] = None):
    """
    Extract per-feature ARD length_scales and convert to importance weights.
    Returns:
        length_scales: (d,) np.ndarray, RBF ARD length scales
        weights:       (d,) np.ndarray, importance ~ exp(-length_scale)
        meta:          Optional dict with labels if f_grid/n_freqs provided
                       - 'kinds': list of 'Re'/'Im' of length d
                       - 'freqs': list of frequencies aligned to features
    Notes:
        If you pass f_grid and n_freqs (where d == 2*n_freqs),
        this will also label the first n_freqs features as Re(f),
        next n_freqs as Im(f), mirroring the paper’s setup.
    """
    fitted_kernel = model_bundle.gp.kernel_
    rbf = _extract_rbf_from_kernel(fitted_kernel)
    length_scales = np.atleast_1d(rbf.length_scale).astype(float)

    # Convert to a monotone "importance" (smaller length_scale => larger weight)
    weights = np.exp(-length_scales)

    meta = None
    if f_grid is not None and n_freqs is not None and len(length_scales) == 2 * n_freqs:
        kinds = np.array(["Re"] * n_freqs + ["Im"] * n_freqs)
        freqs = np.concatenate([f_grid, f_grid])
        meta = {"kinds": kinds, "freqs": freqs}

    return length_scales, weights, meta

# =========================
# (Optional) convenience
# =========================
def summarize_top_features(model_bundle: GPRModelBundle,
                           top_k: int = 10,
                           f_grid: Optional[np.ndarray] = None,
                           n_freqs: Optional[int] = None) -> str:
    """
    Produce a short human-readable summary of the top-K ARD features.
    """
    ls, w, meta = get_ard_importance(model_bundle, f_grid=f_grid, n_freqs=n_freqs)
    idx = np.argsort(-w)[:top_k]
    lines = []
    for i in idx:
        if meta is not None:
            k = meta["kinds"][i]
            f = meta["freqs"][i]
            lines.append(f"{k}@{f:.3f} Hz | weight={w[i]:.6f} | length_scale={ls[i]:.6f}")
        else:
            lines.append(f"feat[{i}] | weight={w[i]:.6f} | length_scale={ls[i]:.6f}")
    return "\n".join(lines)
