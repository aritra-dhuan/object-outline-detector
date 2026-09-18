"""
classical_cv.py
----------------
Classical computer-vision building blocks used both as a fallback (when
YOLO does not detect the requested object) and as supporting analysis
(edge / line / corner statistics) shown in the final report.
"""

import cv2
import numpy as np


def to_gray_blurred(img: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def detect_lines_hough(edges: np.ndarray, min_line_length: int = 40,
                        max_line_gap: int = 10, threshold: int = 50):
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )
    if lines is None:
        return []
    return [tuple(map(int, line)) for line in lines.reshape(-1, 4)]


def detect_harris_corners(gray: np.ndarray, block_size: int = 2, ksize: int = 3,
                           k: float = 0.04, thresh_ratio: float = 0.01):
    response = cv2.cornerHarris(np.float32(gray), block_size, ksize, k)
    response = cv2.dilate(response, None)
    threshold = thresh_ratio * response.max()
    ys, xs = np.where(response > threshold)
    return list(zip(xs.tolist(), ys.tolist())), response


def detect_hessian(gray: np.ndarray, sobel_ksize: int = 3, thresh_ratio: float = 0.01,
                    nms_ksize: int = 9):
    g = np.float32(gray)
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

    det = Ixx * Iyy - Ixy ** 2
    local_max = cv2.dilate(det, np.ones((nms_ksize, nms_ksize), np.uint8))
    threshold = thresh_ratio * max(float(det.max()), 1e-6)
    keep = (det == local_max) & (det > threshold)
    ys, xs = np.where(keep)
    return list(zip(xs.tolist(), ys.tolist())), det


def structural_fallback(img: np.ndarray):
    """Contour-based fallback used when YOLO does not find the requested object."""
    gray = to_gray_blurred(img, 5)
    e = auto_canny(gray)
    kernel = np.ones((5, 5), np.uint8)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    keep = []
    total_area = img.shape[0] * img.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area >= total_area * 0.003:
            keep.append(c)
        if len(keep) >= 8:
            break

    return keep
