const form = document.getElementById("grader-form");
const report = document.getElementById("report");
const fileInput = document.getElementById("blastocyst-image");
const previewImage = document.getElementById("preview-image");
const previewCaption = document.getElementById("preview-caption");
const cohortInput = document.getElementById("cohort");
const ageInput = document.getElementById("age");

let latestPrediction = null;
let latestReport = null;

function apiBase() {
  if (window.location.protocol.startsWith("http")) return "";
  return "http://127.0.0.1:8000";
}

function sanitize(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function showLoading() {
  report.innerHTML = `<p class="muted">Running AI grading and compiling evidence report...</p>`;
}

function showError(message) {
  report.innerHTML = `<p class="highlight"><strong>Error:</strong> ${sanitize(message)}</p>`;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to read image file."));
    reader.readAsDataURL(file);
  });
}

function confidencePct(confidence) {
  const value = Number(confidence || 0);
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

function labelMode(mode) {
  if (mode === "trained_model") return "Trained deep learning model";
  if (mode === "rule_based_cv2_v2") return "OpenCV morphology model";
  return "Rule-based model";
}

function buildAdjustedOddsLine(adjustedOdds) {
  if (!adjustedOdds || typeof adjustedOdds !== "object") return "Not available.";
  const entries = Object.entries(adjustedOdds).map(
    ([key, value]) => `${sanitize(key)} ${sanitize(value)}x`,
  );
  return entries.join(", ");
}

function buildDiagnostics(prediction) {
  const diagnostics = prediction.diagnostics || {};
  const features = prediction.features || {};
  const rows = [
    ["Image quality", diagnostics.image_quality],
    ["Focus", diagnostics.focus_norm],
    ["Contrast", diagnostics.contrast_norm],
    ["Segmentation confidence", diagnostics.segmentation_conf],
    ["Cavity ratio", features.cavity_ratio],
    ["ICM score", features.icm_score],
    ["TE score", features.te_score],
  ];
  return rows
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .map(([label, value]) => {
      const pct = Math.round(Number(value) * 100);
      return `<div class="diag-item"><span>${sanitize(label)}</span><strong>${pct}%</strong></div>`;
    })
    .join("");
}

function buildEvidenceNotes(reportData) {
  const notes = Array.isArray(reportData.evidence_notes)
    ? reportData.evidence_notes
    : [];
  if (!notes.length) return "";
  return `<ul class="evidence-list">${notes
    .map((note) => `<li>${sanitize(note)}</li>`)
    .join("")}</ul>`;
}

function renderReportPanel(prediction, reportData) {
  const confidence = confidencePct(prediction.confidence);
  const outcomeWidth = Math.max(
    4,
    Math.min(96, Number(reportData.live_birth_estimate_pct || 0)),
  );
  const confidenceWidth = Math.max(4, Math.min(96, confidence));
  const overlay = prediction.overlay_png_base64
    ? `<figure class="overlay-figure">
         <img src="data:image/png;base64,${prediction.overlay_png_base64}" alt="Model segmentation overlay" />
         <figcaption>Overlay: yellow embryo boundary, blue cavity, green ICM.</figcaption>
       </figure>`
    : "";

  const ageContext = reportData.age_context
    ? `<p class="highlight">Age context: estimated low-grade trend around <strong>${sanitize(reportData.age_context.estimated_live_birth)}%</strong>. ${sanitize(reportData.age_context.benchmark_points)}</p>`
    : "";

  const warning = prediction.warning
    ? `<p class="highlight"><strong>Fallback note:</strong> ${sanitize(prediction.warning)}</p>`
    : "";

  const explanation = prediction.explanation || {};
  const diagnosticsHtml = buildDiagnostics(prediction);
  const manualTag = reportData.manual_override
    ? `<span class="manual-tag">Manual Override Applied</span>`
    : "";

  return `
    <div class="result-top">
      <span class="grade-chip">${sanitize(reportData.grade || prediction.grade)}</span>
      <span class="band">${sanitize(reportData.quality_label || prediction.quality_band)}</span>
      ${manualTag}
    </div>
    <p><strong>Inference mode:</strong> ${sanitize(labelMode(prediction.mode))}</p>
    <div class="metric">
      <strong>Model confidence:</strong> ${confidence}%
      <div class="bar-shell"><div class="bar confidence-bar" style="width:${confidenceWidth}%"></div></div>
    </div>
    <p><strong>Predicted components:</strong> Stage ${sanitize(prediction.expansion)}, ICM ${sanitize(prediction.icm)}, TE ${sanitize(prediction.te)}</p>
    ${overlay}
    <div class="metric">
      <strong>Estimated live birth rate in selected cohort:</strong> ${sanitize(reportData.live_birth_estimate_pct)}%
      <div class="bar-shell"><div class="bar" style="width:${outcomeWidth}%"></div></div>
    </div>
    <p><strong>Cross-study range:</strong> ${sanitize(reportData.study_range_pct?.low)}% to ${sanitize(reportData.study_range_pct?.high)}%</p>
    <p><strong>Cohort source:</strong> ${sanitize(reportData.cohort_label)}</p>
    <p><strong>Priority signal:</strong> ${sanitize(reportData.priority_signal)}</p>
    <p><strong>Odds context:</strong> ${buildAdjustedOddsLine(reportData.adjusted_odds)}</p>
    <p><strong>Meta-analysis trend:</strong> ${sanitize(reportData.meta_rank_note)}</p>
    <p class="note"><strong>Expansion note:</strong> ${sanitize(reportData.stage_note)}</p>
    <p class="note"><strong>Interpretation:</strong> ${sanitize(reportData.interpretation)}</p>
    <p class="note"><strong>Grade explanation:</strong> ${sanitize(explanation.expansion || "")} ${sanitize(explanation.icm || "")} ${sanitize(explanation.te || "")}</p>
    ${ageContext}
    ${warning}
    <div class="diag-grid">${diagnosticsHtml}</div>
    ${buildEvidenceNotes(reportData)}
    <div class="override-panel">
      <h3>Manual Correction</h3>
      <p class="note">If the AI prediction looks wrong, adjust stage/ICM/TE and regenerate the report from the selected cohort.</p>
      <div class="override-grid">
        <label for="override-stage">Stage</label>
        <select id="override-stage">
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="3">3</option>
          <option value="4">4</option>
          <option value="5">5</option>
          <option value="6">6</option>
        </select>
        <label for="override-icm">ICM</label>
        <select id="override-icm">
          <option value="A">A</option>
          <option value="B">B</option>
          <option value="C">C</option>
          <option value="D">D</option>
        </select>
        <label for="override-te">TE</label>
        <select id="override-te">
          <option value="A">A</option>
          <option value="B">B</option>
          <option value="C">C</option>
          <option value="D">D</option>
        </select>
      </div>
      <button type="button" id="apply-override">Apply Manual Correction</button>
    </div>
  `;
}

async function applyManualOverride() {
  if (!latestPrediction) return;
  const stage = Number(document.getElementById("override-stage").value);
  const icm = document.getElementById("override-icm").value;
  const te = document.getElementById("override-te").value;
  const cohort = cohortInput.value;
  const age = ageInput.value;

  try {
    const payload = { stage, icm, te, cohort };
    if (age) payload.age = Number(age);
    const response = await fetch(`${apiBase()}/api/report-from-grade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Manual correction failed.");
    }
    latestReport = data.report;
    report.innerHTML = renderReportPanel(latestPrediction, latestReport);
    wireOverrideControls();
  } catch (error) {
    showError(error.message);
  }
}

function wireOverrideControls() {
  if (!latestPrediction) return;
  const stageInput = document.getElementById("override-stage");
  const icmInput = document.getElementById("override-icm");
  const teInput = document.getElementById("override-te");
  const button = document.getElementById("apply-override");
  if (!stageInput || !icmInput || !teInput || !button) return;

  stageInput.value = String(
    latestReport && latestReport.manual_override
      ? latestReport.grade.slice(0, 1)
      : latestPrediction.expansion,
  );
  icmInput.value =
    latestReport && latestReport.manual_override
      ? latestReport.grade.slice(1, 2)
      : latestPrediction.icm;
  teInput.value =
    latestReport && latestReport.manual_override
      ? latestReport.grade.slice(2, 3)
      : latestPrediction.te;

  button.onclick = applyManualOverride;
}

function handlePreview() {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    previewCaption.textContent = "No image selected.";
    return;
  }
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.style.display = "block";
  previewCaption.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
}

fileInput.addEventListener("change", handlePreview);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    showError("Please select an image first.");
    return;
  }

  showLoading();

  try {
    const payload = {
      image_data: await fileToDataUrl(file),
      cohort: cohortInput.value,
    };
    if (ageInput.value) payload.age = Number(ageInput.value);

    const response = await fetch(`${apiBase()}/api/grade-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Request failed.");
    }

    latestPrediction = result.prediction;
    latestReport = result.report;
    report.innerHTML = renderReportPanel(latestPrediction, latestReport);
    wireOverrideControls();
  } catch (error) {
    showError(`${error.message} Start the server with: python3 server.py`);
  }
});
