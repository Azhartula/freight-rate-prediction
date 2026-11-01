#!/usr/bin/env python3
"""Train freight-rate models and write submission artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    OUTPUT_DIR,
    TARGET_COLUMN,
    DataPreprocessor,
    contaminated_rpm_mask,
    filter_clean_rates,
    load_december,
    load_template,
    load_train,
    load_validation,
    metrics,
)
from src.december import predict_december
from src.model import (
    compare_models,
    cuda_available,
    fit_full_model,
    predict,
    xgb_device_params,
)
from src.two_part import TwoPartModel

HOLDOUT_START = pd.Timestamp("2025-09-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for prediction CSVs and metrics JSON",
    )
    parser.add_argument(
        "--model",
        choices=[
            "auto",
            "two_part",
            "lightgbm",
            "xgboost",
            "xgboost_rpm",
            "hist_gradient_boosting",
        ],
        default="auto",
        help="Force a model, or auto-select from holdout comparison",
    )
    return parser.parse_args()


def fill_market_from_panels(
    frame: pd.DataFrame, *panels: pd.DataFrame
) -> pd.DataFrame:
    """Fill market_index using day-level medians from available panels."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    pieces = []
    for panel in panels:
        p = panel.copy()
        p["date"] = pd.to_datetime(p["date"])
        if "market_index" in p.columns:
            pieces.append(p[["date", "market_index"]])
    if not pieces:
        return out
    stacked = pd.concat(pieces, ignore_index=True)
    daily = (
        stacked.dropna(subset=["market_index"])
        .assign(date=lambda d: d["date"].dt.normalize())
        .groupby("date")["market_index"]
        .median()
    )
    if "market_index" not in out.columns:
        out["market_index"] = np.nan
    dates = out["date"].dt.normalize()
    out["market_index"] = out["market_index"].fillna(dates.map(daily))
    return out


def evaluate_two_part(
    train_clean: pd.DataFrame, holdout: pd.DataFrame, contam_mask: pd.Series
) -> tuple[TwoPartModel, dict[str, float], dict[str, float]]:
    model = TwoPartModel()
    model.fit(train_clean)
    pred = model.predict(holdout)
    y = holdout[TARGET_COLUMN].to_numpy(dtype=float)
    full_scores = metrics(y, pred)
    clean_scores = metrics(y[~contam_mask.to_numpy()], pred[~contam_mask.to_numpy()])
    return model, full_scores, clean_scores


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_train()
    validation_raw = load_validation()
    december_raw = load_december()
    template = load_template()

    train_fit = train_raw[train_raw["date"] < HOLDOUT_START].copy()
    holdout = train_raw[train_raw["date"] >= HOLDOUT_START].copy()
    print(
        f"Raw train rows: {len(train_fit):,} | Holdout Sep–Oct rows: {len(holdout):,}"
    )

    # Step 1: remove contaminated RPM labels from TRAIN only.
    train_clean, contam_model, clean_stats = filter_clean_rates(train_fit)
    holdout_contam = contaminated_rpm_mask(holdout, contam_model)
    print(
        "RPM contamination filter (equip × distance-bin, train-only): "
        f"removed {clean_stats['contaminated_rows']:,} / {clean_stats['input_rows']:,} "
        f"({clean_stats['contaminated_pct']:.2f}%) "
        f"ratios=[{clean_stats['low_ratio']}, {clean_stats['high_ratio']}]"
    )
    print(
        "Holdout contaminated rows (diagnostic, not dropped): "
        f"{int(holdout_contam.sum()):,} ({100.0 * holdout_contam.mean():.2f}%)"
    )

    device = xgb_device_params()
    print(
        f"XGBoost device: {device.get('device')} "
        f"(CUDA {'available' if cuda_available() else 'NOT available — using CPU'})"
    )

    # Step 2/3: two-part model on cleaned train.
    print("\nFitting two-part model (Ridge level + LightGBM residual)...")
    two_part, two_full, two_clean = evaluate_two_part(
        train_clean, holdout, holdout_contam
    )
    print(
        f"two_part full holdout:  MAE={two_full['mae']:.2f} "
        f"MedAE={two_full['medae']:.2f} MAPE={two_full['mape']:.4f}"
    )
    print(
        f"two_part clean holdout: MAE={two_clean['mae']:.2f} "
        f"MedAE={two_clean['medae']:.2f} MAPE={two_clean['mape']:.4f}"
    )

    # Keep previous GBDT comparison on cleaned train for reference.
    selected_name, summary = compare_models(train_clean, holdout)
    print("\nGBDT comparison on cleaned train → full holdout:")
    print(
        f"{'model':<24} {'target':<8} {'MAE':>8} {'MedAE':>8} "
        f"{'MAPE':>8} {'fit_s':>8}"
    )
    for cand in summary["candidates"]:
        s = cand["scores"]
        print(
            f"{cand['name']:<24} {cand['target_mode']:<8} {s['mae']:8.2f} "
            f"{s['medae']:8.2f} {s['mape']:8.4f} {cand['fit_seconds']:8.1f}"
        )

    best_gbdt = summary["selected_model"]
    best_gbdt_mae = summary["selected_scores"]["mae"]
    two_mae = two_full["mae"]

    if args.model == "auto":
        model_name = "two_part" if two_mae <= best_gbdt_mae else best_gbdt
    else:
        model_name = args.model
    print(
        f"\nSelected: {model_name} "
        f"(two_part MAE={two_mae:.2f} vs best GBDT {best_gbdt} MAE={best_gbdt_mae:.2f})"
    )

    # Final train: clean all Jan–Oct.
    train_full_clean, _, full_clean_stats = filter_clean_rates(train_raw)
    print(
        f"Final clean Jan–Oct rows: {full_clean_stats['clean_rows']:,} "
        f"(removed {full_clean_stats['contaminated_rows']:,})"
    )

    use_two_part = model_name == "two_part"
    if use_two_part:
        print("Fitting final two-part model on cleaned Jan–Oct...")
        final_two = TwoPartModel().fit(train_full_clean)
        # Ensure validation/december have market_index day fills from panels.
        validation_filled = fill_market_from_panels(
            validation_raw, train_raw, validation_raw
        )
        # Need coords on validation already present.
        val_preds = final_two.predict(validation_filled)
        # December: attach coords from clean train city map + MI/quote fills.
        from src.data import DataPreprocessor as DP

        pre = DP().fit(train_full_clean)
        dec = december_raw.copy()
        dec = fill_market_from_panels(dec, train_raw, validation_raw)
        # coords
        pickup_coords = dec["pickup"].map(pre.city_coords_)
        delivery_coords = dec["delivery"].map(pre.city_coords_)
        dec["pickup_lat"] = pickup_coords.apply(lambda x: x[0])
        dec["pickup_lon"] = pickup_coords.apply(lambda x: x[1])
        dec["delivery_lat"] = delivery_coords.apply(lambda x: x[0])
        dec["delivery_lon"] = delivery_coords.apply(lambda x: x[1])
        # quote_signal: Dry Van median from clean train
        qs = float(
            train_full_clean.loc[
                train_full_clean["equipment"] == "Dry Van", "quote_signal"
            ].median()
        )
        dec["quote_signal"] = qs
        dec_preds = final_two.predict(dec)
        december_out = december_raw.copy()
        december_out["predicted_rate"] = np.round(dec_preds, 2)
        model_meta = {
            "name": "two_part",
            "holdout_full": two_full,
            "holdout_clean": two_clean,
        }
    else:
        selected_cand = next(
            c for c in summary["candidates"] if c["name"] == model_name
        )
        if model_name == "hist_gradient_boosting":
            n_estimators = max(int(selected_cand["best_iteration"]), 400)
        else:
            n_estimators = max(int(selected_cand["best_iteration"] * 1.1), 400)
        print(
            f"Fitting final {model_name} on cleaned Jan–Oct "
            f"(n_estimators/max_iter={n_estimators})..."
        )
        model = fit_full_model(
            train_full_clean, model_name, n_estimators=n_estimators
        )
        val_fe = model.preprocessor.transform(validation_raw, include_target=False)
        val_preds = predict(model, val_fe)
        december_out = predict_december(
            train_full_clean,
            model,
            december_raw=december_raw,
            validation_raw=validation_raw,
        )
        model_meta = {
            "name": model_name,
            "n_estimators": n_estimators,
            "holdout": selected_cand["scores"],
        }

    pred_frame = pd.DataFrame(
        {
            "load_id": validation_raw["load_id"].astype(str),
            "predicted_rate": pd.Series(val_preds).round(2),
        }
    )
    ordered = template[["load_id"]].merge(pred_frame, on="load_id", how="left")
    if ordered["predicted_rate"].isna().any():
        raise SystemExit("Missing predictions for one or more template load_id values")
    pred_path = output_dir / "validation_predictions.csv"
    ordered.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path} ({len(ordered):,} rows)")

    december_path = output_dir / "december_chart_inputs.csv"
    december_out.to_csv(december_path, index=False)
    print(f"Wrote {december_path} ({len(december_out)} rows)")

    payload = {
        "selected_model": model_name,
        "contamination_filter": clean_stats,
        "full_train_clean_stats": full_clean_stats,
        "two_part_holdout_full": two_full,
        "two_part_holdout_clean": two_clean,
        "gbdt_summary": summary,
        "model_meta": model_meta,
    }
    metrics_path = output_dir / "holdout_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {metrics_path}")
    print(
        "Done. Next: python score.py "
        "--predictions outputs/validation_predictions.csv "
        "--december-predictions outputs/december_chart_inputs.csv "
        "--output-dir outputs"
    )


if __name__ == "__main__":
    main()
