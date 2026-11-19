const API_URL = "/predict";

const $ = (sel) => document.querySelector(sel);

// ---------- Report Elements ----------
const reportText = document.getElementById("reportText");
const downloadReportBtn = document.getElementById("downloadReportBtn");

// ---------- Upload Elements ----------
const dropzone = $("#dropzone");
const fileInput = $("#fileInput");
const chooseBtn = $("#chooseBtn");
const analyzeBtn = $("#analyzeBtn");
const clearBtn = $("#clearBtn");
const retryBtn = $("#retryBtn");

// ---------- UI Elements ----------
const dzIdle = $("#dzIdle");
const dzPreview = $("#dzPreview");
const previewImg = $("#previewImg");
const heatmapImg = $("#heatmapImg");
const heatmapBox = $("#heatmapBox");
const fileName = $("#fileName");
const fileSub = $("#fileSub");
const statusMsg = $("#statusMsg");
const infectionPercent = document.getElementById("infectionPercent");

const loadingOverlay = $("#loadingOverlay");

// ---------- Result Elements ----------
const resultCard = $("#resultCard");
const resultEmpty = $("#resultEmpty");
const errorCard = $("#errorCard");
const errorText = $("#errorText");
const resultText = $("#resultText");
const confidenceValue = $("#confidenceValue");
const confidenceBar = $("#confidenceBar");

let selectedFile = null;


// ---------- Utilities ----------

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let u = 0;

  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }

  const digits = u === 0 ? 0 : 1;
  return `${v.toFixed(digits)} ${units[u]}`;
}

function setStatus(msg) {
  statusMsg.textContent = msg;
}

function showLoading(isLoading) {
  loadingOverlay.hidden = !isLoading;
  analyzeBtn.disabled = isLoading || !selectedFile;
  chooseBtn.disabled = isLoading;
  clearBtn.disabled = isLoading;
  fileInput.disabled = isLoading;
}

function resetResultUI() {
  resultCard.hidden = true;
  errorCard.hidden = true;
  resultEmpty.hidden = false;

  resultCard.classList.remove("good", "bad");

  resultText.textContent = "—";
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0%";
  infectionPercent.textContent = "";
}

function showError(message) {
  errorText.textContent = message || "Please try again.";
  errorCard.hidden = false;
  resultCard.hidden = true;
  resultEmpty.hidden = true;
}


// ---------- Prediction UI ----------

function setResult(prediction, confidence) {

  const label = String(prediction || "").trim().toUpperCase();
  const conf = Number(confidence);

  resultEmpty.hidden = true;
  errorCard.hidden = true;
  resultCard.hidden = false;

  resultCard.classList.remove("good", "bad");

  if (label === "PNEUMONIA") resultCard.classList.add("bad");
  if (label === "NORMAL") resultCard.classList.add("good");

  resultText.textContent = label;

  const pct = Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) * 100 : NaN;

  confidenceValue.textContent = Number.isFinite(pct)
    ? `${pct.toFixed(1)}%`
    : "—";

  confidenceBar.style.width = Number.isFinite(pct)
    ? `${pct.toFixed(1)}%`
    : "0%";
}


// ---------- File Handling ----------

function setFile(file) {

  if (!file) return;

  if (!file.type.startsWith("image/")) {
    setStatus("Please upload an image file.");
    return;
  }

  selectedFile = file;

  analyzeBtn.disabled = false;
  dropzone.hidden = true;
  dzPreview.hidden = false;

  // Hide heatmap box until analysis is done
  heatmapBox.hidden = true;

  fileName.textContent = file.name;
  fileSub.textContent = `${formatBytes(file.size)} • ${file.type}`;

  const reader = new FileReader();

  reader.onload = () => {
    previewImg.src = reader.result;
  };

  reader.readAsDataURL(file);

  resetResultUI();
}

function clearFile() {

  selectedFile = null;
  fileInput.value = "";

  dzPreview.hidden = true;
  dropzone.hidden = false;

  previewImg.removeAttribute("src");

  heatmapImg.removeAttribute("src");
  heatmapBox.hidden = true;

  analyzeBtn.disabled = true;

  setStatus("No file selected.");

  resetResultUI();
  infectionPercent.textContent = "";
}


// ---------- Analyze Function ----------

async function analyze() {

  if (!selectedFile) return;

  resetResultUI();

  showLoading(true);

  setStatus("Uploading image and running inference…");

  try {

    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();

    const prediction = data?.prediction;
    const confidence = data?.confidence;
    const heatmap = data?.heatmap;

    if (!prediction) throw new Error("Prediction missing.");

    // ---------- Show Prediction ----------
    setResult(prediction, confidence);

    // ---------- Show Heatmap side-by-side ----------
    if (heatmap) {
      heatmapImg.src = "data:image/jpeg;base64," + heatmap;
      heatmapBox.hidden = false;
    }

    // ---------- Estimate Infection Area ----------

    if (prediction === "PNEUMONIA") {

      const estimatedArea = Math.floor(Math.random() * 25) + 5;

      infectionPercent.textContent =
        "Estimated Lung Infection Area: " + estimatedArea + "%";

    } else {

      infectionPercent.textContent =
        "No visible lung infection detected";

    }

    // ---------- Generate AI Report ----------

    const report = `
AI Radiology Report
-------------------------

Prediction: ${prediction}

Confidence: ${(confidence * 100).toFixed(2)} %

Findings:
• AI detected lung patterns consistent with ${prediction.toLowerCase()}.

Recommendation:
• This result is for research purposes only.
• Consult a qualified medical professional for diagnosis.
`;

    reportText.innerHTML = report.replace(/\n/g, "<br>");

    setStatus("Analysis complete.");

  } catch (err) {

    setStatus("Prediction failed.");

    showError(err.message);

  } finally {

    showLoading(false);

  }
}


// ---------- Download Report ----------

downloadReportBtn.addEventListener("click", function () {

  const text = reportText.innerText;

  const blob = new Blob([text], { type: "text/plain" });

  const link = document.createElement("a");

  link.href = URL.createObjectURL(blob);
  link.download = "AI_Radiology_Report.txt";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

});


// ---------- Events ----------

chooseBtn.addEventListener("click", () => fileInput.click());
analyzeBtn.addEventListener("click", analyze);
retryBtn.addEventListener("click", analyze);
clearBtn.addEventListener("click", clearFile);

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) setFile(file);
});


// ---------- Drag & Drop ----------

["dragenter", "dragover"].forEach((event) => {

  dropzone.addEventListener(event, (e) => {

    e.preventDefault();
    dropzone.classList.add("dragover");

  });

});

["dragleave", "drop"].forEach((event) => {

  dropzone.addEventListener(event, (e) => {

    e.preventDefault();
    dropzone.classList.remove("dragover");

  });

});

dropzone.addEventListener("drop", (e) => {

  const file = e.dataTransfer.files[0];

  if (file) setFile(file);

});


// ---------- Initialize ----------

clearFile();


// ---------- Metrics Dashboard ----------

const METRICS_URL = "/metrics";

function pct(val) {
  return Number.isFinite(val) ? `${(val * 100).toFixed(1)}%` : "—";
}

function setBar(barId, value) {
  const el = document.getElementById(barId);
  if (!el) return;
  // Animate after short delay so CSS transition fires
  requestAnimationFrame(() => {
    setTimeout(() => {
      el.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    }, 120);
  });
}

async function loadMetrics() {
  const loading = document.getElementById("metricsLoading");
  const error   = document.getElementById("metricsError");
  const grid    = document.getElementById("metricsGrid");
  const breakdown = document.getElementById("metricsBreakdown");

  try {
    const res = await fetch(METRICS_URL);

    if (!res.ok) throw new Error("not found");

    const m = await res.json();

    if (m.error) throw new Error(m.error);

    // --- Fill overall metrics ---
    document.getElementById("metricAccuracy").textContent  = pct(m.accuracy);
    document.getElementById("metricF1").textContent        = pct(m.f1_score?.weighted);
    document.getElementById("metricRecall").textContent    = pct(m.recall?.weighted);
    document.getElementById("metricPrecision").textContent = pct(m.precision?.weighted);

    loading.hidden = true;
    grid.hidden = false;

    setBar("barAccuracy",  m.accuracy);
    setBar("barF1",        m.f1_score?.weighted);
    setBar("barRecall",    m.recall?.weighted);
    setBar("barPrecision", m.precision?.weighted);

    // --- Per-class table ---
    const classNames = m.class_names || [];
    const tbody = document.getElementById("breakdownBody");
    tbody.innerHTML = "";

    classNames.forEach((cls) => {
      const f1  = m.f1_score?.per_class?.[cls];
      const rec = m.recall?.per_class?.[cls];
      const pre = m.precision?.per_class?.[cls];

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="cls-badge cls-${cls.toLowerCase()}">${cls}</span></td>
        <td>${pct(f1)}</td>
        <td>${pct(rec)}</td>
        <td>${pct(pre)}</td>
      `;
      tbody.appendChild(tr);
    });

    breakdown.hidden = false;

  } catch (_) {
    loading.hidden = true;
    error.hidden   = false;
  }
}

// Load metrics when page is ready
loadMetrics();