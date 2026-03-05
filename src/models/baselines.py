"""
Baseline models — Ridge, ElasticNet, XGBoost.

These are the industry-standard baselines for synergy prediction
(finance-only features). The multimodal deep models should beat these.
"""

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge regression with built-in scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_elasticnet(alpha: float = 1.0, l1_ratio: float = 0.5) -> Pipeline:
    """ElasticNet with built-in scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)),
    ])


def build_xgboost(cfg: dict) -> object:
    """
    Build an XGBoost regressor from config.

    Parameters
    ----------
    cfg : dict
        ``cfg["model"]["xgboost"]`` sub-dict.

    Returns
    -------
    xgboost.XGBRegressor
    """
    from xgboost import XGBRegressor

    xgb_cfg = cfg.get("model", {}).get("xgboost", {})
    return XGBRegressor(
        n_estimators=xgb_cfg.get("n_estimators", 300),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        objective="reg:squarederror",
        random_state=cfg.get("project", {}).get("seed", 42),
        n_jobs=-1,
    )


def build_baseline(cfg: dict) -> object:
    """
    Factory function — return the baseline model specified in config.

    Parameters
    ----------
    cfg : dict
        Full config dictionary (``cfg["model"]["type"]`` selects model).

    Returns
    -------
    sklearn-compatible estimator
    """
    model_type = cfg["model"]["type"]
    if model_type == "ridge":
        alpha = cfg.get("model", {}).get("ridge", {}).get("alpha", 1.0)
        return build_ridge(alpha=alpha)
    elif model_type == "elasticnet":
        return build_elasticnet()
    elif model_type == "xgboost":
        return build_xgboost(cfg)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")
