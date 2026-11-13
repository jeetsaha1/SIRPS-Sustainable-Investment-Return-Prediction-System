"""Ridge demonstration used only to prove the initialization pipeline."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import Ridge


def make_pilot_predictions(dataset: pd.DataFrame, *, alpha: float = 1.0) -> pd.DataFrame:
    """Fit two small Ridge models and return the prediction serving contract."""
    if dataset.empty:
        raise ValueError("The pilot model dataset cannot be empty")
    feature_names = sorted(json.loads(dataset.iloc[0]["features"]))
    features = dataset["features"].map(json.loads).apply(pd.Series)[feature_names]
    predictions: dict[str, object] = {
        "run_id": "pilot-2026",
        "experiment": "A",
        "model": "Ridge",
        "company_id": dataset["company_id"],
        "date": dataset["date"],
        "model_version": "pilot-ridge-1",
        "data_cutoff": dataset["date"].max(),
    }
    for target, output in (("y_excess", "y_hat_excess"), ("y_abs", "y_hat_abs")):
        model = Ridge(alpha=alpha).fit(features, dataset[target])
        predictions[output] = model.predict(features)
    predictions["lower80"] = pd.NA
    predictions["upper80"] = pd.NA
    return pd.DataFrame(predictions)
