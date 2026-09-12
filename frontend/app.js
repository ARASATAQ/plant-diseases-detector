'use strict';

/* ── DOM refs ──────────────────────────────────────────────────────── */
const dropZone    = document.getElementById('dropZone');
const browseLink  = document.getElementById('browseLink');
const fileInput   = document.getElementById('fileInput');
const previewWrap = document.getElementById('previewWrap');
const previewImg  = document.getElementById('previewImg');
const removeBtn   = document.getElementById('removeBtn');
const analyzeBtn  = document.getElementById('analyzeBtn');
const btnText     = document.getElementById('btnText');
const spinner     = document.getElementById('spinner');
const resultPanel = document.getElementById('resultPanel');
const placeholder = document.getElementById('placeholder');
const toast       = document.getElementById('toast');

let selectedFile = null;

/* ── File selection ────────────────────────────────────────────────── */
browseLink.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target !== browseLink) fileInput.click();
});
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

removeBtn.addEventListener('click', resetUpload);

function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image file (JPG, PNG, WEBP).');
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showToast('Image too large — maximum size is 15 MB.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewWrap.style.display = 'block';
    analyzeBtn.disabled = false;
    placeholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  previewWrap.style.display = 'none';
  previewImg.src = '';
  fileInput.value = '';
  analyzeBtn.disabled = true;
  clearResults();
}

/* ── Analyze ───────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', analyzeImage);

async function analyzeImage() {
  if (!selectedFile) return;
  setLoading(true);
  clearResults();

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showToast(`Analysis failed: ${err.message}`);
    placeholder.style.display = '';
    placeholder.textContent = '⚠️ Analysis failed — please try a different image.';
  } finally {
    setLoading(false);
  }
}

/* ── Render results ────────────────────────────────────────────────── */
function renderResults(d) {
  const grade = d.severity_grade;  // Healthy | Mild | Medium | Severe

  const gradeColor = {
    Healthy: 'var(--green)',
    Mild:    'var(--yellow)',
    Medium:  'var(--orange)',
    Severe:  'var(--red)',
  }[grade] || 'var(--muted)';

  const fillClass = {
    Healthy: 'fill-green',
    Mild:    'fill-yellow',
    Medium:  'fill-orange',
    Severe:  'fill-red',
  }[grade] || 'fill-green';

  const emoji = { Healthy: '✅', Mild: '⚠️', Medium: '🔶', Severe: '🚨' }[grade] || '📊';

  // Health score bar width (inverted: higher score = healthier)
  const poiBarW  = Math.min(d.poi, 100).toFixed(1);
  const roiBarW  = Math.min(d.roi_normalized * 100, 100).toFixed(1);
  const scoreBarW = d.severity_score.toFixed(1);

  const modelNote = d.using_fine_tuned_model
    ? '<span style="color:var(--green)">✓ Fine-tuned classifier</span>'
    : '<span style="color:var(--orange)">⚡ Colour estimate — run <code>python download_weights.py</code> for full classifier</span>';

  placeholder.style.display = 'none';

  resultPanel.innerHTML = `
    <!-- ── Grade Card ── -->
    <div class="grade-card">
      <div class="grade-badge ${grade}">
        <span class="score">${Math.round(d.severity_score)}</span>
        <span>${grade.toUpperCase()}</span>
      </div>
      <div class="grade-info">
        <h2 style="color:${gradeColor}">${emoji} ${grade}</h2>
        <div class="dis-name">📌 ${d.disease_class}</div>
        <div class="conf-note">Confidence: ${d.confidence}% &nbsp;·&nbsp; ${modelNote}</div>
      </div>
    </div>

    <!-- ── Metrics Card ── -->
    <div class="metrics-card">
      <div class="card-title">📊 Segmentation Metrics</div>

      <div class="metric-row">
        <div class="metric-label">
          <span>POI — Percentage of Infection</span>
          <span>${d.poi.toFixed(2)} %</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill ${fillClass}" style="width:${poiBarW}%"></div>
        </div>
      </div>

      <div class="metric-row">
        <div class="metric-label">
          <span>ROI — Diseased Pixels</span>
          <span>${d.roi.toLocaleString()} px</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill ${fillClass}" style="width:${roiBarW}%"></div>
        </div>
      </div>

      <div class="metric-row">
        <div class="metric-label">
          <span>Leaf Area (segmented)</span>
          <span>${d.leaf_pixels.toLocaleString()} px</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill fill-green" style="width:100%"></div>
        </div>
      </div>

      <div class="metric-row">
        <div class="metric-label">
          <span>Fuzzy Severity Score (0 = Severe · 100 = Healthy)</span>
          <span>${d.severity_score.toFixed(2)}</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill fill-green" style="width:${scoreBarW}%"></div>
        </div>
      </div>
    </div>

    <!-- ── Overlay Images ── -->
    <div class="images-card">
      <div class="card-title">🔬 Segmentation Overlay</div>
      <div class="images-grid">
        <figure>
          <img src="${d.original_image}" alt="Original leaf" />
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <img src="${d.overlay_image}" alt="Disease overlay" />
          <figcaption>🟢 Healthy tissue &nbsp;·&nbsp; 🔴 Diseased</figcaption>
        </figure>
      </div>
    </div>

    <!-- ── Recommendation ── -->
    <div class="rec-card">
      <span class="rec-icon">💊</span>
      <div class="rec-body">
        <strong>Treatment Recommendation</strong>
        <p>${d.recommendation}</p>
      </div>
    </div>
  `;
}

/* ── Helpers ───────────────────────────────────────────────────────── */
function clearResults() {
  resultPanel.innerHTML = '';
  placeholder.textContent = '📷 Upload a plant leaf image to get started';
  placeholder.style.display = '';
  resultPanel.appendChild(placeholder);
}

function setLoading(on) {
  analyzeBtn.disabled    = on;
  spinner.style.display  = on ? 'block' : 'none';
  btnText.textContent    = on ? 'Analyzing…' : '🔍 Analyze Plant';
}

let toastTimer = null;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4500);
}
