"""Two-part freight rate model: linear market level + LightGBM residual.

Both parts predict log(rate per mile). Final price = distance * exp(part1 + part2).

Part 1 (Ridge) can extrapolate time trend / quarter-end ramp into Nov–Dec.
Part 2 (LightGBM) corrects route/load detail and intentionally has no time features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REFERENCE_DATE = pd.Timestamp("2025-01-01")
QUARTER_RAMP_DAYS = 28
EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
}


def days_to_quarter_end(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(pd.Series(dates))
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (quarter_end - dates.dt.normalize()).dt.days


def quarter_end_ramp(dates: pd.Series) -> pd.Series:
    days_left = days_to_quarter_end(dates)
    return ((QUARTER_RAMP_DAYS - days_left) / QUARTER_RAMP_DAYS).clip(lower=0.0)


def months_since_start(dates: pd.Series) -> pd.Series:
    return (pd.to_datetime(pd.Series(dates)) - REFERENCE_DATE).dt.days / 30.0


def make_target(frame: pd.DataFrame) -> np.ndarray:
    return np.log(
        frame["posted_rate"].astype(float)
        / frame["distance"].astype(float).clip(lower=1.0)
    ).to_numpy(dtype=float)


def to_price(log_rpm: np.ndarray, distance: pd.Series | np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(log_rpm, dtype=float)) * np.asarray(distance, dtype=float), 1.0)


def level_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Part 1: market / trend level features (linear, time-aware)."""
    weight = frame["weight"].astype(float)
    return pd.DataFrame(
        {
            "log_distance": np.log(frame["distance"].astype(float).clip(lower=1.0)),
            "is_reefer": (frame["equipment"] == "Reefer").astype(int),
            "is_flatbed": (frame["equipment"] == "Flatbed").astype(int),
            "weight_10k": weight / 10_000.0,
            "market_index": frame["market_index"].astype(float),
            "months_since_start": months_since_start(frame["date"]).to_numpy(),
            "quarter_end_ramp": quarter_end_ramp(frame["date"]).to_numpy(),
        },
        index=frame.index,
    )


def detail_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Part 2: load/route details (no calendar/time features)."""
    return pd.DataFrame(
        {
            "distance": frame["distance"].astype(float),
            "weight": frame["weight"].astype(float),
            "equipment": frame["equipment"]
            .map(EQUIPMENT_CODES)
            .fillna(0)
            .astype(int),
            "pickup_lat": frame["pickup_lat"].astype(float),
            "pickup_lon": frame["pickup_lon"].astype(float),
            "delivery_lat": frame["delivery_lat"].astype(float),
            "delivery_lon": frame["delivery_lon"].astype(float),
            "delta_lat": frame["delivery_lat"].astype(float)
            - frame["pickup_lat"].astype(float),
            "delta_lon": frame["delivery_lon"].astype(float)
            - frame["pickup_lon"].astype(float),
            "quote_signal": frame["quote_signal"].astype(float),
        },
        index=frame.index,
    )


def prepare_model_frame(frame: pd.DataFrame, *, for_train: bool = False) -> pd.DataFrame:
    """Lightweight input cleaning shared by two-part model."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["weight"] = out["weight"].abs()
    if out["weight"].isna().any():
        out["weight"] = out["weight"].fillna(float(out["weight"].median()))
    if "market_index" not in out.columns:
        out["market_index"] = np.nan
    # Same-date median fill for market_index within the provided frame.
    day = out["date"].dt.normalize()
    day_mi = out.groupby(day)["market_index"].transform("median")
    out["market_index"] = out["market_index"].fillna(day_mi)
    if out["market_index"].isna().any():
        out["market_index"] = out["market_index"].fillna(float(out["market_index"].median()))
    if "quote_signal" not in out.columns:
        out["quote_signal"] = np.nan
    if out["quote_signal"].isna().any():
        out["quote_signal"] = out["quote_signal"].fillna(float(out["quote_signal"].median()))
    return out


@dataclass
class TwoPartModel:
    lgb_params: dict[str, Any] | None = None
    ridge_alpha: float = 1e-3
    part1: Ridge | None = None
    part2: lgb.LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame) -> TwoPartModel:
        df = prepare_model_frame(train, for_train=True)
        y = make_target(df)
        x1 = level_features(df)
        self.part1 = Ridge(alpha=self.ridge_alpha)
        self.part1.fit(x1, y)
        residual = y - self.part1.predict(x1)
        params = {**DEFAULT_LGB_PARAMS, **(self.lgb_params or {})}
        self.part2 = lgb.LGBMRegressor(**params)
        self.part2.fit(detail_features(df), residual)
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        if self.part1 is None or self.part2 is None:
            raise RuntimeError("TwoPartModel must be fit before predict")
        df = prepare_model_frame(frame)
        return self.part1.predict(level_features(df)) + self.part2.predict(
            detail_features(df)
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        df = prepare_model_frame(frame)
        return to_price(self.predict_log_rpm(df), df["distance"])
