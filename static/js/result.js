const CAPTIONS = [
  "Original",
  "Object Silhouette",
  "Canny Edges",
  "Final Object Outline",
  "Black Sketch",
  "Hough Lines",
  "Harris Corners",
  "Hessian Points",
];

const jobId = document.body.dataset.jobId;
const loadingState = document.getElementById("loading-state");
const notfoundState = document.getElementById("notfound-state");
const resultsContent = document.getElementById("results-content");
const heroImg = document.getElementById("hero-img");
const resultsGrid = document.getElementById("results-grid");

let pollTimer = null;

function renderResults(job) {
  heroImg.src = job.original + "?t=" + Date.now();

  resultsGrid.innerHTML = "";
  (job.processed || []).forEach((url, i) => {
    const cell = document.createElement("div");
    cell.className = "grid-cell";

    const img = document.createElement("img");
    img.src = url + "?t=" + Date.now();
    img.alt = CAPTIONS[i] || `Result ${i + 1}`;
    img.loading = "lazy";

    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = CAPTIONS[i] || `Result ${i + 1}`;

    cell.appendChild(img);
    cell.appendChild(caption);
    resultsGrid.appendChild(cell);
  });

  loadingState.hidden = true;
  notfoundState.hidden = true;
  resultsContent.hidden = false;
}

async function fetchJob() {
  try {
    const res = await fetch(`/status/${jobId}`);
    const job = await res.json();

    if (!res.ok) {
      clearInterval(pollTimer);
      loadingState.hidden = true;
      notfoundState.hidden = false;
      return;
    }

    if (job.status === "done") {
      clearInterval(pollTimer);
      renderResults(job);
    }
    // If it's somehow still processing (e.g. direct nav mid-job), keep
    // polling quietly — the loading state stays visible.
  } catch (err) {
    clearInterval(pollTimer);
    loadingState.hidden = true;
    notfoundState.hidden = false;
  }
}

fetchJob();
pollTimer = setInterval(fetchJob, 700);
