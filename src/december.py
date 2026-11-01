"""Complete December chart inputs and generate predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_december, load_validation
from src.model import TrainedModel, predict, predict_quote_signal, train_quote_signal_model


def build_december_features(
    train_raw: pd.DataFrame,
    model: TrainedModel,
    december_raw: pd.DataFrame | None = None,
    validation_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach coords, market_index, and quote_signal to the fixed December lane."""
    december = (december_raw if december_raw is not None else load_december()).copy()
    validation = validation_raw if validation_raw is not None else load_validation()

    # Market index is nearly global by date — use validation feature medians for Dec dates.
    val = validation.dropna(subset=["market_index"]).copy()
    val["_date"] = pd.to_datetime(val["date"]).dt.normalize()
    val_mi = val.groupby("_date")["market_index"].median()
    dates = pd.to_datetime(december["date"]).dt.normalize()
    december["market_index"] = dates.map(val_mi)

    # Fallback via preprocessor date map / nearest for any remaining gaps.
    pre = model.preprocessor
    still_missing = december["market_index"].isna()
    if still_missing.any():
        december.loc[still_missing, "market_index"] = dates[still_missing].map(
            pre.market_by_date_
        )
        if december["market_index"].isna().any():
            december["market_index"] = december["market_index"].fillna(
                pre.global_market_median_
            )

    # Coords from training city lookup.
    pickup_coords = december["pickup"].map(pre.city_coords_)
    delivery_coords = december["delivery"].map(pre.city_coords_)
    december["pickup_lat"] = pickup_coords.apply(lambda x: x[0])
    december["pickup_lon"] = pickup_coords.apply(lambda x: x[1])
    december["delivery_lat"] = delivery_coords.apply(lambda x: x[0])
    december["delivery_lon"] = delivery_coords.apply(lambda x: x[1])

    # Auxiliary quote_signal model.
    qs_pre, qs_encoder, qs_model = train_quote_signal_model(train_raw)
    qs_input = december.copy()
    december["quote_signal"] = predict_quote_signal(
        qs_pre, qs_encoder, qs_model, qs_input
    )

    # Sanity clamp using historical Lexington -> Fort Wayne Dry Van quote_signal.
    hist = train_raw[
        (train_raw["pickup"] == "Lexington")
        & (train_raw["delivery"] == "Fort Wayne")
        & (train_raw["equipment"] == "Dry Van")
    ]["quote_signal"]
    if len(hist):
        lo, hi = float(hist.quantile(0.05)), float(hist.quantile(0.95))
        december["quote_signal"] = december["quote_signal"].clip(lo, hi)

    return december


def predict_december(
    train_raw: pd.DataFrame,
    model: TrainedModel,
    december_raw: pd.DataFrame | None = None,
    validation_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    completed = build_december_features(
        train_raw, model, december_raw=december_raw, validation_raw=validation_raw
    )
    fe = model.preprocessor.transform(completed, include_target=False)
    rates = predict(model, fe)

    out = (december_raw if december_raw is not None else load_december()).copy()
    out["predicted_rate"] = np.round(rates, 2)
    return out
