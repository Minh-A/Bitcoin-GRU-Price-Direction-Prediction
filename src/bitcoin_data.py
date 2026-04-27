"""Shared data utilities for Bitcoin direction modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_with_fallback(path: Path, encoding: str | None = None) -> pd.DataFrame:
    encodings = [encoding] if encoding else []
    encodings.extend(["utf-8-sig", "utf-8", "cp949", "euc-kr"])

    errors: list[str] = []
    for candidate in dict.fromkeys(encodings):
        if not candidate:
            continue
        try:
            return pd.read_csv(path, encoding=candidate)
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate}: {exc}")

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Could not decode {path}. Tried: {', '.join(errors)}",
    )


def load_direction_frame(
    path: Path,
    label_column: str = "Up/Down",
    date_column: str = "date",
    encoding: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    frame = read_csv_with_fallback(path, encoding=encoding)
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    labels = frame[label_column].astype(str).str.strip()
    label_map = {
        "Down": 0,
        "down": 0,
        "DOWN": 0,
        "0": 0,
        "하락": 0,
        "Up": 1,
        "up": 1,
        "UP": 1,
        "1": 1,
        "상승": 1,
    }
    y = labels.map(label_map)
    if y.isna().any():
        unknown = sorted(labels[y.isna()].unique())
        raise ValueError(f"Unknown labels in {label_column}: {unknown}")

    dates = frame[date_column] if date_column in frame.columns else None
    excluded = {label_column}
    if date_column in frame.columns:
        excluded.add(date_column)

    features = frame.drop(columns=[column for column in excluded if column in frame])
    features = features.apply(_to_numeric)
    features = features.dropna(axis=1, how="all")
    features = features.interpolate(method="linear", limit_direction="both")
    features = features.ffill().bfill()

    return features, y.astype(int), dates


def make_sliding_windows(
    features: pd.DataFrame,
    labels: pd.Series,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= window_size:
        raise ValueError("The dataset must contain more rows than window_size.")

    x_values = features.to_numpy(dtype=np.float32)
    y_values = labels.to_numpy(dtype=np.int64)

    windows = []
    targets = []
    for start in range(len(features) - window_size):
        end = start + window_size
        windows.append(x_values[start:end])
        targets.append(y_values[end])

    return np.asarray(windows), np.asarray(targets)


def to_one_hot(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[labels]


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")
