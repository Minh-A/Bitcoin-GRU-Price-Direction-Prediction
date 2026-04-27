"""Rank Bitcoin direction features with tree-based classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

from bitcoin_data import load_direction_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample_real_final_data.csv"))
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--label-column", default="Up/Down")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--model", choices=["rf", "gb"], default="rf")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=100)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    features, labels, _ = load_direction_frame(
        args.data,
        label_column=args.label_column,
        date_column=args.date_column,
        encoding=args.encoding,
    )

    scaled = MinMaxScaler().fit_transform(features)
    x = pd.DataFrame(scaled, columns=features.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    estimator, param_grid = build_search_space(args.model)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=args.n_iter,
        cv=3,
        n_jobs=-1,
        random_state=args.random_state,
    )
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)

    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Best parameters: {search.best_params_}")
    print("Top features:")

    importances = pd.DataFrame(
        {
            "feature": x.columns,
            "importance": search.best_estimator_.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(importances.head(args.top_k).to_string(index=False))


def build_search_space(model_name: str):
    if model_name == "rf":
        return (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": list(range(20, 220, 20)),
                "max_depth": list(range(4, 25, 4)),
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": list(range(2, 14, 2)),
            },
        )

    return (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": list(range(20, 220, 20)),
            "max_depth": list(range(2, 8)),
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_samples_split": list(range(2, 14, 2)),
        },
    )


if __name__ == "__main__":
    main(parse_args())
