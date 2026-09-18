const dropZone = document.getElementById("drop-zone");
const dropZonePrompt = document.getElementById("drop-zone-prompt");
const imageInput = document.getElementById("image-input");
const previewImg = document.getElementById("preview-img");
const clearImgBtn = document.getElementById("clear-img-btn");
const queryInput = document.getElementById("query-input");
const processBtn = document.getElementById("process-btn");
const processBtnFill = document.getElementById("process-btn-fill");
const processBtnLabel = document.getElementById("process-btn-label");
const statusEl = document.getElementById("status");
const errorBox = document.getElementById("error-box");

let selectedFile = null;
let pollTimer = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function setButtonProgress(percent, label) {
  processBtnFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  processBtnLabel.textContent = label;
}

function setBusy(isBusy) {
  if (isBusy) {
    processBtn.disabled = true;
    processBtn.classList.add("is-processing");
    processBtn.classList.remove("is-ready");
  } else {
    processBtn.disabled = !selectedFile;
    processBtn.classList.remove("is-processing");
    setButtonProgress(0, "Process Image");
    if (selectedFile) processBtn.classList.add("is-ready");
  }
}

function handleFile(file) {
  clearError();

  if (!file) return;

  const allowed = ["image/png", "image/jpeg", "image/bmp", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported file type. Please choose a PNG, JPG, JPEG, BMP, or WEBP image.");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropZonePrompt.hidden = true;
    clearImgBtn.hidden = false;
  };
  reader.onerror = () => {
    showError("Could not read the selected file.");
  };
  reader.readAsDataURL(file);

  processBtn.disabled = false;
  processBtn.classList.add("is-ready");
  statusEl.textContent = "";
}

function resetImage() {
  selectedFile = null;
  imageInput.value = "";
  previewImg.hidden = true;
  previewImg.src = "";
  clearImgBtn.hidden = true;
  dropZonePrompt.hidden = false;
  processBtn.disabled = true;
  processBtn.classList.remove("is-ready");
}

dropZone.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  if (imageInput.files && imageInput.files[0]) {
    handleFile(imageInput.files[0]);
  }
});

clearImgBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  resetImage();
});

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFile(file);
});

processBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    showError("Please choose an image first.");
    return;
  }

  clearError();
  setBusy(true);
  setButtonProgress(2, "Uploading...");

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("query", queryInput.value || "");

  try {
    const response = await fetch("/process", {
      method: "POST",
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (parseErr) {
      throw new Error("Server returned an unexpected response.");
    }

    if (!response.ok) {
      throw new Error(data.error || `Request failed with status ${response.status}.`);
    }

    pollJobStatus(data.job_id);
  } catch (err) {
    setBusy(false);
    showError(err.message || "Something went wrong while processing the image.");
  }
});

function pollJobStatus(jobId) {
  clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      const job = await res.json();

      if (!res.ok) {
        throw new Error(job.error || "Lost track of the processing job.");
      }

      if (job.status === "error") {
        clearInterval(pollTimer);
        setBusy(false);
        showError(job.error || "Processing failed.");
        return;
      }

      if (job.status === "done") {
        clearInterval(pollTimer);
        setButtonProgress(100, "Done — opening results...");
        setTimeout(() => {
          window.location.href = `/result/${jobId}`;
        }, 450);
        return;
      }

      const pct = typeof job.percent === "number" ? job.percent : 0;
      setButtonProgress(pct, `${job.message || "Processing"} (${pct}%)`);
    } catch (err) {
      clearInterval(pollTimer);
      setBusy(false);
      showError(err.message || "Lost connection while processing.");
    }
  }, 400);
}


// ---------------------------------------------------------------------
// "About me" corner button — plays a blue, enlarged highlight on click,
// then opens the About page. Without JS it still works as a plain link.
// ---------------------------------------------------------------------
const aboutBtn = document.getElementById("about-btn");

if (aboutBtn) {
  aboutBtn.addEventListener("click", (e) => {
    // let ctrl/cmd/middle-click open in a new tab as normal
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;

    e.preventDefault();
    aboutBtn.classList.add("is-active");

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setTimeout(() => {
      window.location.href = aboutBtn.href;
    }, reduceMotion ? 0 : 320);
  });

  // if the page is restored from the back/forward cache, clear the highlight
  window.addEventListener("pageshow", () => aboutBtn.classList.remove("is-active"));
}
