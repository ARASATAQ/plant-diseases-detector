# 🌿 Plant Disease Detector

Automated plant disease detection and severity grading based on:

> *"Automatic detection and severity analysis of plant disease based on deep learning and fuzzy logic"*

Generalised to **14 plant species** from the PlantVillage dataset.

---

## How It Works

```
Image Upload
     ↓
Phase 1 — Segmentation (no GPU required)
  • GrabCut algorithm  →  leaf vs background mask
  • HSV colour analysis  →  diseased vs healthy tissue pixels
  • Compute ROI (diseased pixel count) + POI (% of leaf infected)
     ↓
Phase 2 — Fuzzy Logic Inference (Mamdani)
  • Inputs : POI (0–100 %) + ROI_normalized (0–1)
  • Rules  : 9 IF-THEN rules from plant pathology
  • Output : severity_score (0–100)
     ↓
Final Grade → Healthy / Mild / Medium / Severe
```

---

## Supported Plants

Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach · Pepper ·
Potato · Raspberry · Soybean · Squash · Strawberry · Tomato

---

## Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU note**: PyTorch automatically uses your **RTX 4060** (CUDA) if available.
> The segmentation + fuzzy pipeline works fine on CPU too.

### 2 — (Optional) Download pre-trained classifier weights
Provides specific disease names (38 PlantVillage classes) instead of colour estimates.
```bash
python download_weights.py
```

### 3 — Start the server
```bash
python run.py
```
Open **http://localhost:8000** in your browser.
Swagger API docs at **http://localhost:8000/docs**.

---

## Makefile Shortcuts

```bash
make install   # pip install -r requirements.txt
make weights   # python download_weights.py
make run       # python run.py
make test      # pytest tests/ -v
```

---

## API Reference

### `POST /api/predict`

Upload a leaf image as `multipart/form-data`. Returns:

```json
{
  "severity_grade":          "Medium",
  "severity_score":          42.3,
  "roi":                     18432,
  "poi":                     28.7,
  "roi_normalized":          0.0448,
  "disease_class":           "Tomato — Early blight",
  "confidence":              83.2,
  "recommendation":          "Apply chlorothalonil or mancozeb fungicide…",
  "overlay_image":           "data:image/png;base64,…",
  "original_image":          "data:image/png;base64,…",
  "leaf_pixels":             411200,
  "disease_pixels":          18432,
  "using_fine_tuned_model":  false
}
```

### `GET /health`
Returns CUDA availability and GPU name.

---

## Project Structure

```
plant-disease-detector/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── routers/predict.py       # POST /predict pipeline
│   ├── models/
│   │   ├── segmentation.py      # GrabCut + HSV disease detection
│   │   ├── classifier.py        # ResNet50 (38-class PlantVillage)
│   │   └── fuzzy_engine.py      # Mamdani fuzzy logic (9 rules)
│   └── utils/
│       ├── image_utils.py       # Preprocess, overlay, base64
│       └── metrics.py           # ROI / POI calculation
├── frontend/
│   ├── index.html               # Drag-drop upload UI
│   ├── style.css                # Dark theme, grade badges
│   └── app.js                   # Fetch API + result rendering
├── weights/                     # ← gitignored; filled by download_weights.py
├── download_weights.py          # Auto-download pre-trained weights
├── run.py                       # Start uvicorn server
├── requirements.txt
└── README.md
```

---

## Fuzzy Logic Rules (Mamdani)

| # | IF poi | AND roi | THEN severity |
|---|--------|---------|---------------|
| 1 | Low    | Small   | Healthy       |
| 2 | Low    | Medium  | Mild          |
| 3 | Low    | Large   | Mild          |
| 4 | Medium | Small   | Mild          |
| 5 | Medium | Medium  | Medium        |
| 6 | Medium | Large   | Medium        |
| 7 | High   | Small   | Medium        |
| 8 | High   | Medium  | Severe        |
| 9 | High   | Large   | Severe        |

Defuzzification: **Centre of Gravity (centroid)** method.

---

## License

MIT
