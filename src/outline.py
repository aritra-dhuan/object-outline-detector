"""
outline.py
----------
Turns a mask (or fallback contours) into the four visual outputs used
by the project: green outline, silhouette, black sketch, and
original+outline overlay.
"""

import cv2
import numpy as np

from .classical_cv import structural_fallback


def get_contours(image: np.ndarray, mask):
    """Return (contours, mode) using the YOLO mask if available, else fallback."""
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 20]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        mode = "AI segmentation"
    else:
        contours = structural_fallback(image)
        mode = "Structural fallback"
    return contours, mode


def build_outputs(image: np.ndarray, contours):
    """Build the green outline, silhouette, and sketch images."""
    final_outline = image.copy()
    cv2.drawContours(final_outline, contours, -1, (0, 255, 0), 4)

    silhouette = np.zeros(image.shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(silhouette, contours, -1, 255, -1)

    sketch = np.full_like(image, 255)
    if contours:
        cv2.drawContours(sketch, contours, -1, (0, 0, 0), 2)

    return final_outline, silhouette, sketch


STYLE_CHOICES = ["green_outline", "black_sketch", "white_silhouette", "original_plus_outline"]


def render_style(style: str, final_outline: np.ndarray, silhouette: np.ndarray, sketch: np.ndarray):
    """Return the (BGR image, title) for a single requested style."""
    if style == "green_outline":
        return final_outline, "Green Object Outline"
    if style == "black_sketch":
        return sketch, "Black Sketch"
    if style == "white_silhouette":
        return cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR), "Object Silhouette"
    if style == "original_plus_outline":
        return final_outline, "Original + Outline"
    raise ValueError(f"Unknown style '{style}'. Choose from {STYLE_CHOICES}")
