"""Model training, multi-model comparison, and prediction helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DataPreprocessor,
    metrics,
)

TargetMode = Literal["absolute", "rpm"]

# Prefer more scalable models when MAE is within this relative margin of the best.
SCALABILITY_MAE_TOLERANCE = 0.02


def cuda_available() -> bool:
    """True when the NVIDIA driver is loaded and XGBoost can see a GPU."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or "GPU" not in (result.stdout or ""):
            return False
    except Exception:
        return False
    # XGBoost wheel in this project is CUDA-enabled; probe device creation lightly.
    try:
        build = xgb.build_info()
        return bool(build.get("USE_CUDA"))
    except Exception:
        return False


def xgb_device_params() -> dict[str, Any]:
    """Use CUDA when the driver is up; otherwise fast CPU hist."""
    if cuda_available():
        return {"device": "cuda", "tree_method": "hist"}
    return {"device": "cpu", "tree_method": "hist"}


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "mae",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_estimators": 2000,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:absoluteerror",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "tree_method": "hist",
    "enable_categorical": False,
    "verbosity": 0,
}

# Holdout-winning RPM configuration (predict $/mile, multiply by distance).
DEFAULT_XGB_RPM_PARAMS: dict[str, Any] = {
    **DEFAULT_XGB_PARAMS,
    "max_depth": 10,
    "min_child_weight": 15,
    "learning_rate": 0.03,
    "n_estimators": 2500,
}

DEFAULT_HGB_PARAMS: dict[str, Any] = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_depth": 8,
    "max_iter": 600,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 30,
    "random_state": 42,
}

# Lower rank = preferred when MAE is nearly tied.
SCALABILITY_RANK = {
    "lightgbm": 1,
    "xgboost": 2,
    "xgboost_rpm": 2,
    "hist_gradient_boosting": 3,
}


@dataclass
class CategoryEncoder:
    """Map categorical strings to stable integer codes for all model backends."""

    maps_: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> CategoryEncoder:
        self.maps_ = {}
        for col in CATEGORICAL_FEATURES:
            values = sorted(frame[col].astype(str).unique().tolist())
            mapping = {value: idx for idx, value in enumerate(values)}
            if "UNK" not in mapping:
                mapping["UNK"] = len(mapping)
            self.maps_[col] = mapping
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in CATEGORICAL_FEATURES:
            mapping = self.maps_[col]
            unk = mapping.get("UNK", -1)
            out[col] = out[col].astype(str).map(mapping).fillna(unk).astype(int)
        return out


@dataclass
class TrainedModel:
    name: str
    preprocessor: DataPreprocessor
    encoder: CategoryEncoder
    estimator: Any
    feature_columns: list[str]
    best_iteration: int
    holdout_metrics: dict[str, float]
    target_mode: TargetMode = "absolute"
    fit_seconds: float = 0.0


def _matrix(frame: pd.DataFrame, encoder: CategoryEncoder) -> pd.DataFrame:
    encoded = encoder.transform(frame)
    return encoded[FEATURE_COLUMNS].copy()


def _target(frame: pd.DataFrame, target_mode: TargetMode) -> pd.Series:
    y = frame[TARGET_COLUMN].astype(float)
    if target_mode == "rpm":
        return y / frame["distance"].astype(float).clip(lower=1.0)
    return y


def baseline_quote_distance(frame: pd.DataFrame) -> np.ndarray:
    return (frame["quote_signal"].astype(float) * frame["distance"].astype(float)).to_numpy()


def baseline_equip_rpm(frame: pd.DataFrame) -> np.ndarray:
    return (frame["equip_mean_rpm"].astype(float) * frame["distance"].astype(float)).to_numpy()


def evaluate_baselines(holdout: pd.DataFrame) -> dict[str, dict[str, float]]:
    y = holdout[TARGET_COLUMN].to_numpy()
    return {
        "quote_signal_x_distance": metrics(y, baseline_quote_distance(holdout)),
        "equipment_mean_rpm_x_distance": metrics(y, baseline_equip_rpm(holdout)),
    }


def predict_estimator(
    estimator: Any,
    frame: pd.DataFrame,
    encoder: CategoryEncoder,
    target_mode: TargetMode = "absolute",
) -> np.ndarray:
    x = _matrix(frame, encoder)
    preds = np.asarray(estimator.predict(x), dtype=float)
    if target_mode == "rpm":
        preds = preds * frame["distance"].astype(float).to_numpy()
    return np.maximum(preds, 1.0)


def predict(model: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    return predict_estimator(
        model.estimator, frame, model.encoder, target_mode=model.target_mode
    )


def _cat_feature_indices() -> list[int]:
    return [FEATURE_COLUMNS.index(col) for col in CATEGORICAL_FEATURES]


def train_lightgbm(
    train_df: pd.DataFrame,
    encoder: CategoryEncoder,
    valid_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    target_mode: TargetMode = "absolute",
) -> tuple[Any, int]:
    cfg = {**DEFAULT_LGBM_PARAMS, **(params or {}), "random_state": seed}
    n_estimators = int(cfg.pop("n_estimators"))
    model = lgb.LGBMRegressor(**cfg, n_estimators=n_estimators)

    x_train = _matrix(train_df, encoder)
    y_train = _target(train_df, target_mode)
    fit_kwargs: dict[str, Any] = {"categorical_feature": _cat_feature_indices()}
    if valid_df is not None and len(valid_df):
        x_valid = _matrix(valid_df, encoder)
        y_valid = _target(valid_df, target_mode)
        fit_kwargs["eval_X"] = x_valid
        fit_kwargs["eval_y"] = y_valid
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    model.fit(x_train, y_train, **fit_kwargs)
    best_iteration = getattr(model, "best_iteration_", None) or n_estimators
    return model, int(best_iteration)


def train_xgboost(
    train_df: pd.DataFrame,
    encoder: CategoryEncoder,
    valid_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    target_mode: TargetMode = "absolute",
) -> tuple[Any, int]:
    base = DEFAULT_XGB_RPM_PARAMS if target_mode == "rpm" else DEFAULT_XGB_PARAMS
    cfg = {**base, **xgb_device_params(), **(params or {}), "random_state": seed}
    n_estimators = int(cfg.pop("n_estimators"))
    model = xgb.XGBRegressor(**cfg, n_estimators=n_estimators)

    x_train = _matrix(train_df, encoder)
    y_train = _target(train_df, target_mode)

    fit_kwargs: dict[str, Any] = {}
    if valid_df is not None and len(valid_df):
        x_valid = _matrix(valid_df, encoder)
        y_valid = _target(valid_df, target_mode)
        fit_kwargs["eval_set"] = [(x_valid, y_valid)]
        fit_kwargs["verbose"] = False
        model.set_params(early_stopping_rounds=100)

    model.fit(x_train, y_train, **fit_kwargs)
    best_iteration = int(getattr(model, "best_iteration", None) or n_estimators)
    return model, best_iteration


def train_hist_gradient(
    train_df: pd.DataFrame,
    encoder: CategoryEncoder,
    valid_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    target_mode: TargetMode = "absolute",
) -> tuple[Any, int]:
    cfg = {**DEFAULT_HGB_PARAMS, **(params or {})}
    if valid_df is not None and len(valid_df):
        cfg["early_stopping"] = False
        cfg.pop("validation_fraction", None)
        cfg.pop("n_iter_no_change", None)
    model = HistGradientBoostingRegressor(
        **cfg,
        categorical_features=_cat_feature_indices(),
    )
    x_train = _matrix(train_df, encoder)
    y_train = _target(train_df, target_mode)
    model.fit(x_train, y_train)
    best_iteration = int(getattr(model, "n_iter_", cfg.get("max_iter", 600)))
    return model, best_iteration


def _candidate(
    name: str,
    scores: dict[str, float],
    fit_seconds: float,
    best_iteration: int,
    params: dict[str, Any],
    target_mode: TargetMode,
) -> dict[str, Any]:
    return {
        "name": name,
        "scores": scores,
        "fit_seconds": fit_seconds,
        "best_iteration": best_iteration,
        "params": params,
        "target_mode": target_mode,
        "scalability_rank": SCALABILITY_RANK.get(name, 99),
    }


def compare_models(
    train_raw: pd.DataFrame,
    holdout_raw: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    """Compare absolute and RPM-target models; pick best accuracy/scalability tradeoff."""
    pre = DataPreprocessor().fit(train_raw)
    train_fe = pre.transform(train_raw, include_target=True)
    holdout_fe = pre.transform(holdout_raw, include_target=True)
    encoder = CategoryEncoder().fit(train_fe)

    baseline_scores = evaluate_baselines(holdout_fe)
    y_holdout = holdout_fe[TARGET_COLUMN].to_numpy()
    candidates: list[dict[str, Any]] = []

    # 1) LightGBM absolute
    t0 = time.perf_counter()
    model, iters = train_lightgbm(train_fe, encoder, holdout_fe)
    secs = time.perf_counter() - t0
    preds = predict_estimator(model, holdout_fe, encoder)
    candidates.append(
        _candidate(
            "lightgbm",
            metrics(y_holdout, preds),
            secs,
            iters,
            {
                k: DEFAULT_LGBM_PARAMS[k]
                for k in ("num_leaves", "min_child_samples", "learning_rate", "lambda_l2")
            },
            "absolute",
        )
    )

    # 2) XGBoost absolute
    t0 = time.perf_counter()
    model, iters = train_xgboost(train_fe, encoder, holdout_fe, target_mode="absolute")
    secs = time.perf_counter() - t0
    preds = predict_estimator(model, holdout_fe, encoder, target_mode="absolute")
    candidates.append(
        _candidate(
            "xgboost",
            metrics(y_holdout, preds),
            secs,
            iters,
            {
                k: DEFAULT_XGB_PARAMS[k]
                for k in ("max_depth", "min_child_weight", "learning_rate", "reg_lambda")
            },
            "absolute",
        )
    )

    # 3) XGBoost RPM (predict $/mile * distance) — strongest MAE in experiments
    t0 = time.perf_counter()
    model, iters = train_xgboost(train_fe, encoder, holdout_fe, target_mode="rpm")
    secs = time.perf_counter() - t0
    preds = predict_estimator(model, holdout_fe, encoder, target_mode="rpm")
    candidates.append(
        _candidate(
            "xgboost_rpm",
            metrics(y_holdout, preds),
            secs,
            iters,
            {
                k: DEFAULT_XGB_RPM_PARAMS[k]
                for k in ("max_depth", "min_child_weight", "learning_rate", "reg_lambda")
            },
            "rpm",
        )
    )

    # 4) HistGradientBoosting absolute
    t0 = time.perf_counter()
    model, iters = train_hist_gradient(train_fe, encoder, holdout_fe)
    secs = time.perf_counter() - t0
    preds = predict_estimator(model, holdout_fe, encoder)
    candidates.append(
        _candidate(
            "hist_gradient_boosting",
            metrics(y_holdout, preds),
            secs,
            iters,
            {
                k: DEFAULT_HGB_PARAMS[k]
                for k in ("max_depth", "learning_rate", "max_iter", "l2_regularization")
            },
            "absolute",
        )
    )

    best_mae = min(c["scores"]["mae"] for c in candidates)
    eligible = [
        c
        for c in candidates
        if c["scores"]["mae"] <= best_mae * (1.0 + SCALABILITY_MAE_TOLERANCE)
    ]
    winner = sorted(eligible, key=lambda c: (c["scalability_rank"], c["scores"]["mae"]))[0]

    summary = {
        "baselines": baseline_scores,
        "candidates": candidates,
        "selection_rule": (
            f"lowest MAE; if within {SCALABILITY_MAE_TOLERANCE:.0%} of best MAE, "
            "prefer LightGBM > XGBoost/XGBoost-RPM > HistGradientBoosting"
        ),
        "selected_model": winner["name"],
        "selected_target_mode": winner["target_mode"],
        "selected_scores": winner["scores"],
        "selected_params": winner["params"],
        "selected_best_iteration": winner["best_iteration"],
        "selected_fit_seconds": winner["fit_seconds"],
    }
    return winner["name"], summary


def fit_full_model(
    train_raw: pd.DataFrame,
    model_name: str,
    n_estimators: int | None = None,
    seed: int = 42,
) -> TrainedModel:
    pre = DataPreprocessor().fit(train_raw)
    train_fe = pre.transform(train_raw, include_target=True)
    encoder = CategoryEncoder().fit(train_fe)

    target_mode: TargetMode = "rpm" if model_name == "xgboost_rpm" else "absolute"
    t0 = time.perf_counter()
    if model_name == "lightgbm":
        params = {"n_estimators": n_estimators or 1200}
        estimator, best_iteration = train_lightgbm(
            train_fe, encoder, valid_df=None, params=params, seed=seed
        )
    elif model_name in {"xgboost", "xgboost_rpm"}:
        params = {"n_estimators": n_estimators or 1200}
        estimator, best_iteration = train_xgboost(
            train_fe,
            encoder,
            valid_df=None,
            params=params,
            seed=seed,
            target_mode=target_mode,
        )
    elif model_name == "hist_gradient_boosting":
        params = {"max_iter": n_estimators or 600, "early_stopping": False}
        estimator, best_iteration = train_hist_gradient(
            train_fe, encoder, valid_df=None, params=params
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    fit_seconds = time.perf_counter() - t0

    return TrainedModel(
        name=model_name,
        preprocessor=pre,
        encoder=encoder,
        estimator=estimator,
        feature_columns=list(FEATURE_COLUMNS),
        best_iteration=best_iteration,
        holdout_metrics={},
        target_mode=target_mode,
        fit_seconds=fit_seconds,
    )


def train_quote_signal_model(
    train_raw: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[DataPreprocessor, CategoryEncoder, lgb.LGBMRegressor]:
    """Auxiliary LightGBM to impute quote_signal for December rows."""
    pre = DataPreprocessor().fit(train_raw)
    fe = pre.transform(train_raw, include_target=False)
    qs_features = [
        c
        for c in FEATURE_COLUMNS
        if c not in {"quote_signal", "qs_x_distance", "mi_x_qs_x_distance"}
    ]
    encoder = CategoryEncoder().fit(fe)
    x = encoder.transform(fe)[qs_features].copy()
    y = train_raw["quote_signal"].astype(float)

    cfg = {
        **DEFAULT_LGBM_PARAMS,
        **(params or {}),
        "objective": "mae",
        "n_estimators": 800,
        "random_state": 42,
    }
    n_estimators = int(cfg.pop("n_estimators"))
    model = lgb.LGBMRegressor(**cfg, n_estimators=n_estimators)
    cat_idx = [qs_features.index(c) for c in CATEGORICAL_FEATURES if c in qs_features]
    model.fit(x, y, categorical_feature=cat_idx)
    return pre, encoder, model


def predict_quote_signal(
    pre: DataPreprocessor,
    encoder: CategoryEncoder,
    model: lgb.LGBMRegressor,
    frame_raw: pd.DataFrame,
) -> np.ndarray:
    fe = pre.transform(frame_raw, include_target=False)
    qs_features = [
        c
        for c in FEATURE_COLUMNS
        if c not in {"quote_signal", "qs_x_distance", "mi_x_qs_x_distance"}
    ]
    x = encoder.transform(fe)[qs_features].copy()
    return np.asarray(model.predict(x), dtype=float)
