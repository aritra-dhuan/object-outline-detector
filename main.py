#!/usr/bin/env python3
"""
Object Outline Detector — Flask web app
=========================================

A local web UI around the same YOLO11 segmentation + classical
computer-vision (Canny/Hough/Harris/Hessian) pipeline used by the
original command-line tool. All processing logic lives unchanged in
src/segmentation.py, src/classical_cv.py, src/outline.py, and
src/report.py — this file only adds a Flask server and browser UI
around it.

Run
---
    python3 main.py

On first run, this script automatically installs any missing Python
dependencies (Flask, OpenCV, NumPy, Ultralytics/YOLO) into the current
environment — no separate `pip install -r requirements.txt` step is
required. It then starts a local server at http://127.0.0.1:5000 and
automatically opens it in your default web browser.
"""

import importlib
import os
import subprocess
import sys
import threading
import uuid
import webbrowser


# ---------------------------------------------------------------------------
# Dependency bootstrap — runs before any third-party imports below, so a
# completely bare Python environment can still start this app with just
# `python3 main.py`.
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    # (module name to try importing, pip package spec to install if missing)
    ("flask", "flask>=3.0.0"),
    ("werkzeug", "werkzeug>=3.0.0"),
    ("cv2", "opencv-python-headless>=4.9.0"),
    ("numpy", "numpy>=1.24.0"),
    ("ultralytics", "ultralytics>=8.3.0"),
]

# Guards against re-exec looping forever if packages genuinely can't be
# made importable (see ensure_dependencies below).
_BOOTSTRAP_RETRY_ENV_VAR = "_OOD_BOOTSTRAP_RETRIED"


def _importable(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _pip_install(pip_specs):
    """Install `pip_specs` with pip, retrying with --break-system-packages
    if the environment is an "externally managed" one (e.g. recent
    Debian/Ubuntu system Python) that would otherwise refuse the install."""
    base_cmd = [sys.executable, "-m", "pip", "install", "--quiet",
                "--disable-pip-version-check", *pip_specs]

    result = subprocess.run(base_cmd, capture_output=True, text=True)
    if result.returncode != 0 and "externally-managed-environment" in (result.stderr or ""):
        result = subprocess.run(base_cmd + ["--break-system-packages"],
                                 capture_output=True, text=True)
    return result


def ensure_dependencies():
    missing_specs = [spec for name, spec in REQUIRED_PACKAGES if not _importable(name)]

    if not missing_specs:
        return

    print(f"Using Python interpreter: {sys.executable}")
    print("First-time setup: installing missing Python packages...")
    for spec in missing_specs:
        print(f"  - {spec}")
    print("This may take a few minutes (larger packages like ultralytics/opencv take longest)...")

    result = _pip_install(missing_specs)
    if result.returncode != 0:
        print("\nAutomatic dependency installation failed.", file=sys.stderr)
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        print("\nPlease install manually and re-run:", file=sys.stderr)
        print(f"    {sys.executable} -m pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    # pip reported success, but that doesn't guarantee the packages are
    # importable *in this already-running process* — e.g. pip can fall
    # back to installing into the user site-packages directory (when it
    # lacks permission for the system one), and a directory that didn't
    # exist yet when this interpreter started isn't always picked up
    # mid-run. Re-check, and if anything is still not importable here,
    # restart the script as a brand-new process, which reliably re-scans
    # the real environment.
    importlib.invalidate_caches()
    try:
        import site
        user_site = site.getusersitepackages()
        if os.path.isdir(user_site) and user_site not in sys.path:
            site.addsitedir(user_site)
    except Exception:
        pass

    still_missing = [name for name, _ in REQUIRED_PACKAGES if not _importable(name)]

    if still_missing:
        if os.environ.get(_BOOTSTRAP_RETRY_ENV_VAR) == "1":
            # Already restarted once for this exact reason — don't loop forever.
            print(
                "\nPackages were installed by pip but are still not importable "
                "in this Python interpreter after restarting:",
                file=sys.stderr,
            )
            print("  " + ", ".join(still_missing), file=sys.stderr)
            print(f"\nInterpreter: {sys.executable}", file=sys.stderr)
            print(
                "\nThis usually means there are multiple Python installations "
                "on this machine and pip installed the packages for a "
                "different one than main.py is running under. Try installing "
                "directly with this exact interpreter:\n"
                f"    {sys.executable} -m pip install -r requirements.txt\n"
                "or run main.py inside a virtual environment:\n"
                "    python3 -m venv venv && source venv/bin/activate\n"
                "    pip install -r requirements.txt\n"
                "    python3 main.py",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Packages installed, but not yet visible to this running process.")
        print("Restarting to pick them up...\n")
        os.environ[_BOOTSTRAP_RETRY_ENV_VAR] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    print("Dependencies installed successfully.\n")


ensure_dependencies()

# --- Third-party imports (guaranteed available after the bootstrap above) ---
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from src.segmentation import load_model, segment_object, clean_mask
from src.classical_cv import (
    to_gray_blurred,
    auto_canny,
    detect_lines_hough,
    detect_harris_corners,
    detect_hessian,
)
from src.outline import get_contours, build_outputs
from src.report import print_analysis

# ---------------------------------------------------------------------------
# Paths (all relative to this file — no hardcoded absolute paths)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
HOST = "127.0.0.1"
PORT = 5000
MODEL_NAME = "yolo11n-seg.pt"
CONF_THRESHOLD = 0.25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# The YOLO model is loaded once, lazily, on first request (not at import
# time) so that `python3 main.py` starts the server instantly and opens
# the browser without waiting on model download/load.
_model_lock = threading.Lock()
_model = None

# ---------------------------------------------------------------------------
# In-memory job tracking — lets the browser poll real processing progress
# (percentage complete) instead of just spinning blindly, and lets the
# results page be a separate URL the browser is sent to once a job finishes.
# ---------------------------------------------------------------------------
_jobs_lock = threading.Lock()
JOBS = {}


def _update_job(job_id, **fields):
    with _jobs_lock:
        job = JOBS.setdefault(job_id, {})
        job.update(fields)


def _get_job(job_id):
    with _jobs_lock:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            print(f"Loading model: {MODEL_NAME} ...")
            _model = load_model(MODEL_NAME)
    return _model


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_results():
    """Delete every processed image left in RESULTS_DIR by earlier uploads
    (the .gitkeep placeholder is kept), and forget the old jobs that
    pointed at them, so only the newest upload's outputs ever exist."""
    for name in os.listdir(RESULTS_DIR):
        if name == ".gitkeep":
            continue
        path = os.path.join(RESULTS_DIR, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
        except OSError as exc:
            print(f"Could not delete old result {name}: {exc}")

    with _jobs_lock:
        JOBS.clear()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/process", methods=["POST"])
def process():
    # --- Validate the incoming request ---
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    query = request.form.get("query", "").strip().lower()

    # --- Save the upload under a unique, sanitized name ---
    ext = file.filename.rsplit(".", 1)[1].lower()
    job_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(f"{job_id}.{ext}")
    upload_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        file.save(upload_path)
    except Exception as exc:
        return jsonify({"error": f"Could not save uploaded file: {exc}"}), 500

    # --- Read the image ---
    image = cv2.imread(upload_path)
    if image is None:
        return jsonify({"error": "Uploaded file could not be read as an image."}), 400

    # --- The upload is valid: wipe the previous upload's processed images
    # so the results/ folder only ever holds the current run. ---
    clear_results()

    # --- Kick the actual pipeline off in a background thread and return
    # immediately, so the browser can poll /status/<job_id> for real
    # progress and drive an animated percentage on the Process button. ---
    original_url = f"/uploads/{safe_name}"
    _update_job(job_id, status="processing", percent=0, message="Starting...")

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, image, query, original_url),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>")
def status(job_id):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id."}), 404
    return jsonify(job)


@app.route("/result/<job_id>")
def result_page(job_id):
    return render_template("result.html", job_id=job_id)


def _run_job(job_id: str, image: np.ndarray, query: str, original_url: str):
    """Runs run_pipeline in a background thread and records progress/result
    into the JOBS dict for /status/<job_id> to report back."""

    def report(percent, message):
        _update_job(job_id, status="processing", percent=percent, message=message)

    try:
        result_files = run_pipeline(image, query, job_id, progress_cb=report)
        processed_urls = [f"/results/{fname}" for fname in result_files]
        _update_job(
            job_id,
            status="done",
            percent=100,
            message="Done.",
            original=original_url,
            processed=processed_urls,
        )
    except Exception as exc:
        _update_job(job_id, status="error", error=f"Processing failed: {exc}")


def run_pipeline(image: np.ndarray, object_name: str, job_id: str, progress_cb=None):
    """
    Runs the existing YOLO + classical-CV pipeline on `image` and saves
    the 8 processed output images into RESULTS_DIR, prefixed with job_id.

    `progress_cb`, if given, is called as progress_cb(percent, message) at
    each stage so callers (e.g. the /process route) can report real-time
    progress back to the browser.

    Returns the list of saved filenames, in display order.
    """

    def tick(percent, message):
        if progress_cb:
            progress_cb(percent, message)

    tick(5, "Loading model...")
    model = get_model()

    mask = None
    if object_name:
        tick(15, f"Looking for '{object_name}'...")
        mask, confidence, detected_label = segment_object(
            model, image, object_name, conf=CONF_THRESHOLD
        )
        if mask is not None:
            mask = clean_mask(mask)
            print(f"Detected: {detected_label} (confidence {confidence:.2%})")
        else:
            print(f"'{object_name}' not found by YOLO. Using structural fallback.")
    else:
        tick(15, "No query given — using structural fallback...")
        print("No object query provided. Using structural fallback.")

    tick(32, "Extracting contours...")
    contours, mode = get_contours(image, mask)
    final_outline, silhouette, sketch = build_outputs(image, contours)

    tick(48, "Detecting edges (Canny)...")
    gray = to_gray_blurred(image)
    edges = auto_canny(gray)

    tick(62, "Detecting lines (Hough)...")
    lines = detect_lines_hough(edges)

    tick(75, "Detecting corners (Harris)...")
    harris_points, _ = detect_harris_corners(gray)

    tick(85, "Detecting blobs (Hessian)...")
    hessian_points, _ = detect_hessian(gray)

    print_analysis(image, mode, silhouette, lines, harris_points, hessian_points)

    hough_vis = image.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    harris_vis = image.copy()
    for x, y in harris_points:
        cv2.circle(harris_vis, (x, y), 3, (0, 0, 255), -1)

    hessian_vis = image.copy()
    for x, y in hessian_points:
        cv2.circle(hessian_vis, (x, y), 3, (255, 0, 255), -1)

    tick(93, "Saving results...")

    # The exact 8 outputs the original CLI script saved via cv2.imwrite
    # (same filenames/order, just prefixed per job so concurrent uploads
    # don't collide).
    outputs = [
        ("01_original", image),
        ("02_object_silhouette", silhouette),
        ("03_canny_edges", edges),
        ("04_final_object_outline", final_outline),
        ("05_black_sketch", sketch),
        ("06_hough_lines", hough_vis),
        ("07_harris_corners", harris_vis),
        ("08_hessian_points", hessian_vis),
    ]

    saved_files = []
    for label, img in outputs:
        fname = f"{job_id}_{label}.png"
        path = os.path.join(RESULTS_DIR, fname)
        cv2.imwrite(path, img)
        saved_files.append(fname)

    tick(99, "Finalizing...")

    return saved_files


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    # Open the browser shortly after the server starts, but only in the
    # actual running process (not the Werkzeug reloader's parent process).
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, open_browser).start()

    app.run(host=HOST, port=PORT, debug=True, use_reloader=True)
