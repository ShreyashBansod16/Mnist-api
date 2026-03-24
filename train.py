import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils_mnist import load_mnist_idx_dataset


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a production-ready MNIST digit classifier")
    parser.add_argument("--data-root", type=str, default=".", help="Path containing IDX files")
    parser.add_argument("--output-dir", type=str, default="model_artifacts", help="Model output folder")
    parser.add_argument("--epochs", type=int, default=18, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation split size")
    args = parser.parse_args()

    set_deterministic(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist_idx_dataset(args.data_root)

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    val_size = min(max(1000, args.val_size), x_train.shape[0] // 2)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train_fit, y_train_fit = x_train[:-val_size], y_train[:-val_size]

    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]

    model.fit(
        x_train_fit,
        y_train_fit,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    keras_model_path = output_dir / "digit_cnn.keras"
    model.save(keras_model_path)

    saved_model_path = output_dir / "saved_model"
    model.export(saved_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    (output_dir / "digit_cnn.tflite").write_bytes(tflite_model)

    metadata = {
        "input_shape": [28, 28, 1],
        "classes": list(range(10)),
        "test_accuracy": float(test_acc),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved Keras model: {keras_model_path}")
    print(f"Saved TF SavedModel: {saved_model_path}")
    print(f"Saved TFLite model: {output_dir / 'digit_cnn.tflite'}")


if __name__ == "__main__":
    main()
