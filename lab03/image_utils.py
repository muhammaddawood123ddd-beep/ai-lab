"""Laboratory 03: Image Processing and Feature Engineering Utilities.

Author: Muhammad Dawood
Course: Introduction to Artificial Intelligence
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union


def image_to_features(path: str, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Load an image from disk, convert to grayscale, resize to fixed dimensions,

    and flatten into a 1D feature vector.

    Parameters
    ----------
    path : str
        Local filesystem path to an image file.
    size : tuple of (int, int), default=(64, 64)
        Target resolution (width, height) for feature standardization.

    Returns
    -------
    np.ndarray
        A 1D float32 NumPy array containing the flattened pixel intensities.
        Length is guaranteed to be size[0] * size[1].
    """
    with Image.open(path) as img:
        img_gray = img.convert("L")
        img_resized = img_gray.resize(size)
        features = np.array(img_resized, dtype=np.float32).flatten()
        return features


if __name__ == "__main__":
    print("Testing image_to_features...")
    import os
    test_path = "data/sample_photo.png"
    if os.path.exists(test_path):
        feats = image_to_features(test_path)
        print(f"Loaded {test_path} -> feature vector shape: {feats.shape}")
        assert feats.shape == (4096,), "Expected 4096 features for 64x64 image"
        print("Test passed successfully!")
    else:
        print(f"Sample image not found at {test_path}")
