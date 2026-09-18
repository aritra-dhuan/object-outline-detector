"""
report.py
---------
Prints the final object-outline analysis summary to the console.
"""

import cv2


def print_analysis(image, mode, silhouette, lines, harris_points, hessian_points):
    image_area = image.shape[0] * image.shape[1]
    object_area = int(cv2.countNonZero(silhouette))
    boundary_pixels = int(cv2.countNonZero(cv2.Canny(silhouette, 50, 150)))
    object_ratio = (object_area / image_area * 100) if image_area else 0

    print("=" * 42)
    print("       OBJECT OUTLINE ANALYSIS")
    print("=" * 42)
    print(f"Image size       : {image.shape[1]} x {image.shape[0]}")
    print(f"Processing mode  : {mode}")
    print(f"Object area      : {object_area:,} pixels")
    print(f"Object coverage  : {object_ratio:.2f}%")
    print(f"Boundary pixels  : {boundary_pixels:,}")
    print(f"Hough lines      : {len(lines)}")
    print(f"Harris corners   : {len(harris_points)}")
    print(f"Hessian points   : {len(hessian_points)}")
    print("=" * 42)

    return {
        "image_size": (image.shape[1], image.shape[0]),
        "mode": mode,
        "object_area_px": object_area,
        "object_coverage_pct": round(object_ratio, 2),
        "boundary_pixels": boundary_pixels,
        "hough_lines": len(lines),
        "harris_corners": len(harris_points),
        "hessian_points": len(hessian_points),
    }
