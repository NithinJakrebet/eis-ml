"""Model back-ends with one shared shape.

Every module here exposes::

    fit(X, y, **params) -> bundle
    predict(bundle, X)  -> (mean, std)

so experiments can swap models by name via ``MODELS[name]``.
"""

from . import gpr, xgb

MODELS = {"gpr": gpr, "xgb": xgb}

__all__ = ["MODELS", "gpr", "xgb"]
