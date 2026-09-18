# Object Outline Detector — Web App

A local Flask web app around the YOLO11 + classical computer-vision
(Canny/Hough/Harris/Hessian) object-outline pipeline. Upload an image,
type what you want to find, click **Process**, and see the original
image plus all 8 processed outputs in a 3×3 grid.

All detection/segmentation logic is unchanged from the original
script — it lives in `src/segmentation.py`, `src/classical_cv.py`,
`src/outline.py`, and `src/report.py`. `main.py` only adds the Flask
server and web UI around it.

## Project Structure

```
object-outline-detector/
├── main.py                 # Flask app + pipeline runner (start this)
├── requirements.txt
├── src/
│   ├── segmentation.py      # YOLO model loading + mask extraction
│   ├── classical_cv.py      # Canny / Hough / Harris / Hessian + fallback
│   ├── outline.py           # Contour extraction + outline rendering
│   └── report.py            # Console analysis summary
├── templates/
│   ├── index.html           # Upload UI (with the "About me" corner button)
│   ├── result.html          # Results grid
│   └── about.html           # About me + project details page
├── static/
│   ├── css/style.css
│   └── js/script.js         # Upload, preview, AJAX processing, grid render
├── uploads/                 # Uploaded images are saved here at runtime
├── results/                 # Processed output images are saved here at runtime
└── notebooks/
    └── original_colab_prototype.ipynb   # Original exploratory notebook (reference only)
```

`results/` is cleared automatically on every new upload, so it only ever holds
the latest run's output images.

`uploads/` and `results/` are created automatically if missing, and are
git-ignored except for a `.gitkeep` placeholder, so the folders exist
in a fresh clone even though their contents aren't committed.

## Requirements

- Python 3.9+
- pip
- Internet access on first run, to install missing Python packages automatically and to download the YOLO model weights (`yolo11n-seg.pt`, ~6 MB)

## Setup

```bash
git clone https://github.com/<your-username>/object-outline-detector.git
cd object-outline-detector
```

That's it for setup — you do **not** need to run `pip install` yourself.
`main.py` checks for its required packages (Flask, OpenCV, NumPy,
Ultralytics) on startup and automatically installs any that are
missing before the server starts. `requirements.txt` is still included
for reference or for anyone who prefers to install manually ahead of
time with `pip install -r requirements.txt`.

A virtual environment is still recommended so the auto-installed
packages don't affect your global Python install:

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

## Run

```bash
python3 main.py
```

This starts the Flask server at `http://127.0.0.1:5000` and
automatically opens it in your default web browser. On the very first
run it will print progress while it installs any missing packages —
this can take a few minutes (OpenCV and Ultralytics are the largest).
Subsequent runs start instantly. If the browser doesn't open
automatically, just navigate to `http://127.0.0.1:5000` yourself.

## Using the app

1. Click the upload area (or drag an image onto it) — a preview appears.
2. Type the object you're looking for in **"What do you want to find?"**
   (e.g. `car`, `dog`, `person`). Leave it blank to force the
   structural (non-AI) fallback outline.
3. Click **Process**.
4. Once processing finishes, a 3×3 grid appears with:
   1. The original uploaded image
   2. Object silhouette
   3. Canny edges
   4. Final object outline (green)
   5. Black sketch
   6. Hough lines
   7. Harris corners
   8. Hessian interest points

Errors (no file selected, unsupported file type, unreadable image,
processing failure) are shown in an on-page error message rather than
failing silently.

## Notes

- On startup, `main.py` checks each required package by trying to
  import it; anything missing is installed via
  `pip install <package>` (with an automatic `--break-system-packages`
  retry for "externally managed" system Python installs). If the
  installed packages aren't immediately visible to the running process
  (this can happen when pip falls back to installing into your user
  site-packages directory), the script automatically restarts itself
  once to pick them up. If they're still not importable after that, it
  stops with a clear diagnostic (showing the exact interpreter path)
  instead of retrying forever — usually the fix in that case is to
  install and run with the *same* Python (e.g. inside a venv):
  ```bash
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  python3 main.py
  ```
- The YOLO model is loaded lazily, on the first `/process` request —
  not at startup — so `python3 main.py` opens the browser quickly
  rather than waiting on model load.
- Model weights (`*.pt`) are not committed to the repository; they are
  downloaded automatically by `ultralytics` on first use.
- All file paths in `main.py` are computed relative to the script's own
  location (`os.path.dirname(os.path.abspath(__file__))`), so the app
  works regardless of the directory it's launched from.
- Uploaded files are validated by extension (`png`, `jpg`, `jpeg`,
  `bmp`, `webp`) and capped at 16 MB.
