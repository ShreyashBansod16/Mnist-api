import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_idx_path(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {candidates}")


def read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic number: {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, rows, cols)
    return images


def read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic number: {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != num_labels:
        raise ValueError(
            f"Label count mismatch in {path}: header={num_labels}, actual={labels.shape[0]}"
        )
    return labels


def load_mnist_idx_dataset(root_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root_dir)

    train_images_path = _resolve_idx_path(
        root,
        [
            "train-images.idx3-ubyte",
            "train-images-idx3-ubyte/train-images-idx3-ubyte",
        ],
    )
    train_labels_path = _resolve_idx_path(
        root,
        [
            "train-labels.idx1-ubyte",
            "train-labels-idx1-ubyte/train-labels-idx1-ubyte",
        ],
    )
    test_images_path = _resolve_idx_path(
        root,
        [
            "t10k-images.idx3-ubyte",
            "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
        ],
    )
    test_labels_path = _resolve_idx_path(
        root,
        [
            "t10k-labels.idx1-ubyte",
            "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
        ],
    )

    x_train = read_idx_images(train_images_path)
    y_train = read_idx_labels(train_labels_path)
    x_test = read_idx_images(test_images_path)
    y_test = read_idx_labels(test_labels_path)

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training image/label count mismatch")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test image/label count mismatch")

    return x_train, y_train, x_test, y_test


def preprocess_canvas_image(image_bytes: bytes) -> np.ndarray:
    """Convert a user-drawn canvas image into a normalized MNIST-style tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(image, dtype=np.float32) / 255.0

    # If background is white with dark strokes, invert to MNIST style (light digit on dark background).
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = np.clip(arr, 0.0, 1.0)
    mask = arr > 0.1
    if not np.any(mask):
        raise ValueError("No visible digit found in the uploaded image")

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    cropped = arr[y_min : y_max + 1, x_min : x_max + 1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resample = getattr(Image, "Resampling", Image).LANCZOS
    resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize((new_w, new_h), resample)
    resized_arr = np.array(resized, dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_arr

    # Match training-time normalization shape: (batch, height, width, channel).
    return canvas.reshape(1, 28, 28, 1).astype(np.float32)
