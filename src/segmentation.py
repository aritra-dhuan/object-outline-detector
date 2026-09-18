"""
segmentation.py
----------------
Loads the YOLO11 segmentation model and produces a binary mask for a
user-requested object class found in an image.
"""

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_NAME = "yolo11n-seg.pt"

# Common aliases users might type instead of the exact COCO class name.
CLASS_ALIASES = {
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "tv": "tv",
    "cellphone": "cell phone",
    "mobile": "cell phone",
}


def load_model(model_name: str = MODEL_NAME) -> YOLO:
    """Load (and, on first run, download) the YOLO segmentation model."""
    model = YOLO(model_name)
    return model


def find_matching_class_id(model: YOLO, name: str):
    """Resolve a user-provided object name to a YOLO class id, if any."""
    name = name.lower().strip()
    name = CLASS_ALIASES.get(name, name)

    for idx, label in model.names.items():
        if label.lower() == name:
            return idx
    return None


def segment_object(model: YOLO, image: np.ndarray, object_name: str, conf: float = 0.25):
    """
    Run YOLO segmentation on `image` and merge all mask instances that
    match `object_name`.

    Returns:
        mask (np.ndarray | None): binary (0/255) mask, or None if no match
        confidence (float | None): highest confidence among matched instances
        detected_label (str | None): the resolved YOLO class label
    """
    class_id = find_matching_class_id(model, object_name)

    result = model.predict(source=image, conf=conf, verbose=False)[0]

    mask = None
    confidence = None
    detected_label = None

    if class_id is not None and result.masks is not None and result.boxes is not None:
        for i, cls_tensor in enumerate(result.boxes.cls):
            cls = int(cls_tensor.item())
            if cls == class_id:
                c = float(result.boxes.conf[i].item())
                candidate = result.masks.data[i].cpu().numpy()
                candidate = cv2.resize(
                    candidate,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                if mask is None:
                    mask = candidate
                    confidence = c
                    detected_label = model.names[cls]
                else:
                    mask = np.maximum(mask, candidate)
                    if c > confidence:
                        confidence = c

    if mask is not None:
        mask = (mask > 0.5).astype(np.uint8) * 255

    return mask, confidence, detected_label


def clean_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Morphological close + open to remove noise while keeping all instances."""
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned
