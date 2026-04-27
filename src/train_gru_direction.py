"""Train a GRU model for Bitcoin daily price-direction prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from bitcoin_data import load_direction_frame, make_sliding_windows, to_one_hot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample_real_final_data.csv"))
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--label-column", default="Up/Down")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--test-size", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def build_model(input_shape: tuple[int, int], seed: int):
    import tensorflow as tf

    tf.random.set_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.GRU(
                32,
                input_shape=input_shape,
                activation="tanh",
                return_sequences=True,
            ),
            tf.keras.layers.GRU(64, activation="tanh", return_sequences=True),
            tf.keras.layers.GRU(128, activation="tanh", return_sequences=True),
            tf.keras.layers.GRU(256, activation="tanh", return_sequences=True),
            tf.keras.layers.GRU(512, activation="tanh"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(args: argparse.Namespace) -> None:
    features, labels, _ = load_direction_frame(
        args.data,
        label_column=args.label_column,
        date_column=args.date_column,
        encoding=args.encoding,
    )

    if args.test_size <= args.window_size:
        raise ValueError("test_size must be larger than window_size.")
    if len(features) <= args.test_size + args.window_size:
        raise ValueError("Dataset is too small for the requested test_size/window_size.")

    train_features = features.iloc[: -args.test_size].reset_index(drop=True)
    test_features = features.iloc[-args.test_size :].reset_index(drop=True)
    train_labels = labels.iloc[: -args.test_size].reset_index(drop=True)
    test_labels = labels.iloc[-args.test_size :].reset_index(drop=True)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    train_scaled_frame = train_features.copy()
    test_scaled_frame = test_features.copy()
    train_scaled_frame.loc[:, :] = train_scaled
    test_scaled_frame.loc[:, :] = test_scaled

    x_train, y_train = make_sliding_windows(
        train_scaled_frame,
        train_labels,
        window_size=args.window_size,
    )
    x_test, y_test = make_sliding_windows(
        test_scaled_frame,
        test_labels,
        window_size=args.window_size,
    )

    model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]), seed=args.seed)
    history = model.fit(
        x_train,
        to_one_hot(y_train),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"Train windows: {len(x_train):,}")
    print(f"Test windows: {len(x_test):,}")
    print(f"Features: {x_train.shape[2]:,}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Down", "Up"],
            zero_division=0,
        )
    )

    if args.plot_dir:
        save_plots(args.plot_dir, history.history, y_test, predictions)


def save_plots(
    plot_dir: Path,
    history: dict[str, list[float]],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("accuracy", []), label="accuracy", color="red")
    plt.plot(epochs, history.get("loss", []), label="loss", color="blue")
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "training_history.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    x_axis = np.arange(len(actual))
    plt.scatter(x_axis, actual, label="actual", s=120, color="gray", alpha=0.7)
    plt.scatter(x_axis, predicted, label="prediction", s=35, color="red")
    plt.yticks([0, 1], ["Down", "Up"])
    plt.ylim(-0.5, 1.5)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "prediction_scatter.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    train(parse_args())
