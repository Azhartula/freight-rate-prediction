"""Data loading, cleaning, and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
NUMERIC_FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "log_distance",
    "inv_distance",
    "qs_x_distance",
    "mi_x_qs_x_distance",
    "delta_lat",
    "delta_lon",
    "bearing",
    "haversine_miles",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "doy_sin",
    "doy_cos",
    "equip_mean_rpm",
    "lane_mean_rpm",
    "pickup_mean_rpm",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "posted_rate"

# Contaminated labels sit outside a clear gap vs typical RPM for the same
# equipment and distance band (~1.4% of rows). Remove from training only.
RPM_CONTAM_LOW_RATIO = 0.5
RPM_CONTAM_HIGH_RATIO = 1.8
RPM_CONTAM_DISTANCE_BINS = 20


def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return frame["posted_rate"].astype(float) / frame["distance"].astype(float).clip(
        lower=1.0
    )


@dataclass
class RpmContaminationModel:
    """Typical RPM by (equipment, distance-bin), fit on a training fold only."""

    distance_edges_: np.ndarray | None = None
    typical_rpm_: pd.Series | None = None
    low_ratio: float = RPM_CONTAM_LOW_RATIO
    high_ratio: float = RPM_CONTAM_HIGH_RATIO
    n_bins: int = RPM_CONTAM_DISTANCE_BINS

    def fit(self, frame: pd.DataFrame) -> RpmContaminationModel:
        rpm = rate_per_mile(frame)
        edges = np.quantile(
            frame["distance"].astype(float), np.linspace(0.0, 1.0, self.n_bins + 1)
        )
        edges = edges.astype(float)
        edges[0], edges[-1] = 0.0, np.inf
        # Deduplicate edges if quantiles collide.
        edges = np.unique(edges)
        if edges[-1] != np.inf:
            edges = np.append(edges, np.inf)
        self.distance_edges_ = edges
        keys = [
            frame["equipment"].astype(str),
            pd.cut(frame["distance"].astype(float), bins=edges, include_lowest=True),
        ]
        self.typical_rpm_ = rpm.groupby(keys, observed=True).median()
        return self

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        if self.distance_edges_ is None or self.typical_rpm_ is None:
            raise RuntimeError("RpmContaminationModel must be fit before mask()")
        rpm = rate_per_mile(frame)
        keys = pd.MultiIndex.from_arrays(
            [
                frame["equipment"].astype(str),
                pd.cut(
                    frame["distance"].astype(float),
                    bins=self.distance_edges_,
                    include_lowest=True,
                ),
            ]
        )
        typical = self.typical_rpm_.reindex(keys).astype(float)
        # Fallback: equipment-level median of typical table, then global.
        if typical.isna().any():
            equip_fallback = self.typical_rpm_.groupby(level=0).median()
            typical = typical.fillna(frame["equipment"].astype(str).map(equip_fallback))
            typical = typical.fillna(float(self.typical_rpm_.median()))
        ratio = rpm.to_numpy(dtype=float) / typical.to_numpy(dtype=float)
        return pd.Series(
            (ratio < self.low_ratio) | (ratio > self.high_ratio), index=frame.index
        )


def fit_equipment_rpm_medians(frame: pd.DataFrame) -> dict[str, float]:
    """Backward-compatible helper: equipment median RPM."""
    tmp = frame.copy()
    tmp["_rpm"] = rate_per_mile(tmp)
    return tmp.groupby("equipment")["_rpm"].median().astype(float).to_dict()


def contaminated_rpm_mask(
    frame: pd.DataFrame,
    contamination_model: RpmContaminationModel | dict[str, float],
    *,
    low_ratio: float = RPM_CONTAM_LOW_RATIO,
    high_ratio: float = RPM_CONTAM_HIGH_RATIO,
) -> pd.Series:
    """True where posted_rate looks like label contamination."""
    if isinstance(contamination_model, RpmContaminationModel):
        return contamination_model.mask(frame)
    # Legacy dict[str, float] equipment medians.
    rpm = rate_per_mile(frame)
    med = frame["equipment"].map(contamination_model).astype(float)
    global_med = (
        float(np.median(list(contamination_model.values())))
        if contamination_model
        else 2.15
    )
    med = med.fillna(global_med)
    ratio = rpm / med.clip(lower=1e-6)
    return (ratio < low_ratio) | (ratio > high_ratio)


def filter_clean_rates(
    frame: pd.DataFrame,
    contamination_model: RpmContaminationModel | None = None,
    *,
    low_ratio: float = RPM_CONTAM_LOW_RATIO,
    high_ratio: float = RPM_CONTAM_HIGH_RATIO,
) -> tuple[pd.DataFrame, RpmContaminationModel, dict[str, float]]:
    """Drop contaminated RPM rows. Fit contamination model from `frame` if needed."""
    model = contamination_model
    if model is None:
        model = RpmContaminationModel(
            low_ratio=low_ratio, high_ratio=high_ratio
        ).fit(frame)
    mask = model.mask(frame)
    clean = frame.loc[~mask].copy()
    stats = {
        "input_rows": int(len(frame)),
        "contaminated_rows": int(mask.sum()),
        "clean_rows": int((~mask).sum()),
        "contaminated_pct": float(100.0 * mask.mean()),
        "low_ratio": float(model.low_ratio),
        "high_ratio": float(model.high_ratio),
    }
    return clean, model, stats


def load_validation(path: Path = VALIDATION_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_december(path: Path = DECEMBER_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_template(path: Path = TEMPLATE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _haversine_miles(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 3958.8 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def _smoothed_mean(
    mean: float, count: float, global_mean: float, prior_strength: float
) -> float:
    """Shrink group mean toward global mean: (n*mean + m*global) / (n+m)."""
    return float(
        (count * mean + prior_strength * global_mean) / (count + prior_strength)
    )


@dataclass
class DataPreprocessor:
    """Fit cleaning statistics and regularized encodings on a training fold only."""

    # Winsorization / clip bounds fitted on train (1st–99th percentile).
    clip_bounds_: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Smoothing strength for target encodings (higher = more shrinkage).
    # Tuned on Sep–Oct holdout; 8 keeps MAE near the unregularized RPM model.
    encoding_prior_: float = 8.0
    weight_medians_: dict[str, float] = field(default_factory=dict)
    global_weight_median_: float = 0.0
    market_by_date_: dict[pd.Timestamp, float] = field(default_factory=dict)
    global_market_median_: float = 1.0
    city_coords_: dict[str, tuple[float, float]] = field(default_factory=dict)
    equip_mean_rpm_: dict[str, float] = field(default_factory=dict)
    lane_mean_rpm_: dict[tuple[str, str], float] = field(default_factory=dict)
    pickup_mean_rpm_: dict[str, float] = field(default_factory=dict)
    global_mean_rpm_: float = 2.15
    known_pickups_: set[str] = field(default_factory=set)
    known_deliveries_: set[str] = field(default_factory=set)
    known_equipment_: set[str] = field(default_factory=set)

    def fit(self, frame: pd.DataFrame) -> DataPreprocessor:
        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["weight"] = df["weight"].abs()

        # Fit clip bounds before imputation uses medians.
        self.clip_bounds_ = {}
        # Clip noisy sensors/signals only. Do not clip distance: RPM models
        # multiply predictions by distance, so the true miles must be preserved.
        for col, series in [
            ("weight", df["weight"].dropna()),
            ("market_index", df["market_index"].dropna()),
            ("quote_signal", df["quote_signal"].dropna()),
        ]:
            lo, hi = np.quantile(series.to_numpy(dtype=float), [0.01, 0.99])
            self.clip_bounds_[col] = (float(lo), float(hi))

        weight_for_stats = df["weight"].clip(*self.clip_bounds_["weight"])
        self.global_weight_median_ = float(weight_for_stats.median())
        self.weight_medians_ = (
            df.assign(weight=weight_for_stats)
            .groupby("equipment")["weight"]
            .median()
            .astype(float)
            .to_dict()
        )
        self.global_market_median_ = float(
            df["market_index"].clip(*self.clip_bounds_["market_index"]).median()
        )
        self.market_by_date_ = (
            df.assign(
                market_index=df["market_index"].clip(*self.clip_bounds_["market_index"])
            )
            .groupby(df["date"].dt.normalize())["market_index"]
            .median()
            .dropna()
            .astype(float)
            .to_dict()
        )

        pickup_coords = (
            df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first().to_dict("index")
        )
        delivery_coords = (
            df.groupby("delivery")[["delivery_lat", "delivery_lon"]]
            .first()
            .to_dict("index")
        )
        self.city_coords_ = {}
        for city, vals in pickup_coords.items():
            self.city_coords_[city] = (float(vals["pickup_lat"]), float(vals["pickup_lon"]))
        for city, vals in delivery_coords.items():
            self.city_coords_.setdefault(
                city, (float(vals["delivery_lat"]), float(vals["delivery_lon"]))
            )

        # Clip extreme RPM labels when building encodings so rare wild rates shrink less noisily.
        distance = df["distance"].astype(float).clip(lower=1.0)
        rpm = (df["posted_rate"] / distance).clip(
            lower=np.quantile(df["posted_rate"] / distance, 0.01),
            upper=np.quantile(df["posted_rate"] / distance, 0.99),
        )
        self.global_mean_rpm_ = float(rpm.mean())
        prior = self.encoding_prior_

        equip_stats = (
            df.assign(_rpm=rpm).groupby("equipment")["_rpm"].agg(["mean", "count"])
        )
        self.equip_mean_rpm_ = {
            idx: _smoothed_mean(
                float(row["mean"]), float(row["count"]), self.global_mean_rpm_, prior
            )
            for idx, row in equip_stats.iterrows()
        }

        lane_stats = (
            df.assign(_rpm=rpm)
            .groupby(["pickup", "delivery"])["_rpm"]
            .agg(["mean", "count"])
        )
        # Stronger prior for lanes (rarer than equipment/pickup).
        lane_prior = prior * 1.5
        self.lane_mean_rpm_ = {
            (idx[0], idx[1]): _smoothed_mean(
                float(row["mean"]), float(row["count"]), self.global_mean_rpm_, lane_prior
            )
            for idx, row in lane_stats.iterrows()
        }

        pickup_stats = (
            df.assign(_rpm=rpm).groupby("pickup")["_rpm"].agg(["mean", "count"])
        )
        self.pickup_mean_rpm_ = {
            idx: _smoothed_mean(
                float(row["mean"]), float(row["count"]), self.global_mean_rpm_, prior
            )
            for idx, row in pickup_stats.iterrows()
        }

        self.known_pickups_ = set(df["pickup"].astype(str))
        self.known_deliveries_ = set(df["delivery"].astype(str))
        self.known_equipment_ = set(df["equipment"].astype(str))
        return self

    def _clip_series(self, series: pd.Series, column: str) -> pd.Series:
        if column not in self.clip_bounds_:
            return series.astype(float)
        lo, hi = self.clip_bounds_[column]
        return series.astype(float).clip(lower=lo, upper=hi)

    def _impute_weight(self, series: pd.Series, equipment: pd.Series) -> pd.Series:
        values = series.abs()
        out = values.copy()
        missing = out.isna()
        if missing.any():
            filled = equipment.map(self.weight_medians_).astype(float)
            out = out.where(~missing, filled)
            out = out.fillna(self.global_weight_median_)
        return self._clip_series(out, "weight")

    def _impute_market_index(self, series: pd.Series, dates: pd.Series) -> pd.Series:
        out = series.astype(float).copy()
        missing = out.isna()
        if missing.any():
            date_vals = pd.to_datetime(dates).dt.normalize()
            mapped = date_vals.map(self.market_by_date_).astype(float)
            out = out.where(~missing, mapped)
            # Fill remaining with nearest known date median, else global.
            if out.isna().any():
                known = (
                    pd.Series(self.market_by_date_)
                    .sort_index()
                    .rename("market_index")
                )
                if not known.empty:
                    refill_idx = out[out.isna()].index
                    nearest = []
                    for idx in refill_idx:
                        d = date_vals.loc[idx]
                        pos = known.index.get_indexer([d], method="nearest")[0]
                        nearest.append(float(known.iloc[pos]))
                    out.loc[refill_idx] = nearest
                out = out.fillna(self.global_market_median_)
        return self._clip_series(out, "market_index")

    def transform(
        self, frame: pd.DataFrame, *, include_target: bool = False
    ) -> pd.DataFrame:
        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"])

        if "pickup_lat" not in df.columns or df["pickup_lat"].isna().any():
            coords = df["pickup"].map(self.city_coords_)
            df["pickup_lat"] = df.get("pickup_lat", pd.Series(index=df.index, dtype=float))
            df["pickup_lon"] = df.get("pickup_lon", pd.Series(index=df.index, dtype=float))
            miss = df["pickup_lat"].isna()
            if miss.any():
                df.loc[miss, "pickup_lat"] = coords[miss].apply(
                    lambda x: x[0] if isinstance(x, tuple) else np.nan
                )
                df.loc[miss, "pickup_lon"] = coords[miss].apply(
                    lambda x: x[1] if isinstance(x, tuple) else np.nan
                )

        if "delivery_lat" not in df.columns or df["delivery_lat"].isna().any():
            coords = df["delivery"].map(self.city_coords_)
            df["delivery_lat"] = df.get(
                "delivery_lat", pd.Series(index=df.index, dtype=float)
            )
            df["delivery_lon"] = df.get(
                "delivery_lon", pd.Series(index=df.index, dtype=float)
            )
            miss = df["delivery_lat"].isna()
            if miss.any():
                df.loc[miss, "delivery_lat"] = coords[miss].apply(
                    lambda x: x[0] if isinstance(x, tuple) else np.nan
                )
                df.loc[miss, "delivery_lon"] = coords[miss].apply(
                    lambda x: x[1] if isinstance(x, tuple) else np.nan
                )

        df["weight"] = self._impute_weight(df["weight"], df["equipment"])
        if "market_index" not in df.columns:
            df["market_index"] = np.nan
        df["market_index"] = self._impute_market_index(df["market_index"], df["date"])

        if "quote_signal" not in df.columns:
            df["quote_signal"] = np.nan
        else:
            df["quote_signal"] = self._clip_series(df["quote_signal"], "quote_signal")

        df["pickup"] = df["pickup"].astype(str).where(
            df["pickup"].astype(str).isin(self.known_pickups_), "UNK"
        )
        df["delivery"] = df["delivery"].astype(str).where(
            df["delivery"].astype(str).isin(self.known_deliveries_), "UNK"
        )
        df["equipment"] = df["equipment"].astype(str).where(
            df["equipment"].astype(str).isin(self.known_equipment_), "UNK"
        )

        distance = df["distance"].astype(float).clip(lower=1.0)
        df["distance"] = distance
        df["log_distance"] = np.log1p(distance)
        df["inv_distance"] = 1.0 / distance
        df["qs_x_distance"] = df["quote_signal"].astype(float) * distance
        df["mi_x_qs_x_distance"] = (
            df["market_index"].astype(float) * df["quote_signal"].astype(float) * distance
        )
        df["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
        df["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]
        df["bearing"] = _bearing(
            df["pickup_lat"].to_numpy(),
            df["pickup_lon"].to_numpy(),
            df["delivery_lat"].to_numpy(),
            df["delivery_lon"].to_numpy(),
        )
        df["haversine_miles"] = _haversine_miles(
            df["pickup_lat"].to_numpy(),
            df["pickup_lon"].to_numpy(),
            df["delivery_lat"].to_numpy(),
            df["delivery_lon"].to_numpy(),
        )

        doy = df["date"].dt.dayofyear.astype(float)
        df["dayofweek"] = df["date"].dt.dayofweek.astype(int)
        df["month"] = df["date"].dt.month.astype(int)
        df["dayofyear"] = doy.astype(int)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        df["equip_mean_rpm"] = (
            df["equipment"].map(self.equip_mean_rpm_).fillna(self.global_mean_rpm_)
        )
        lane_keys = list(zip(df["pickup"], df["delivery"]))
        df["lane_mean_rpm"] = [
            self.lane_mean_rpm_.get(key, self.global_mean_rpm_) for key in lane_keys
        ]
        df["pickup_mean_rpm"] = (
            df["pickup"].map(self.pickup_mean_rpm_).fillna(self.global_mean_rpm_)
        )

        cols = list(FEATURE_COLUMNS)
        if include_target and TARGET_COLUMN in df.columns:
            cols = cols + [TARGET_COLUMN]
        if "load_id" in df.columns:
            cols = ["load_id"] + cols
        cols = ["date"] + [c for c in cols if c != "date"]
        return df[cols]

    def fit_transform(
        self, frame: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        return self.fit(frame).transform(frame, include_target=include_target)


def metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "mae": float(err.mean()),
        "medae": float(np.median(err)),
        "mape": float((err / np.clip(np.abs(y_true), 1e-6, None)).mean()),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }
