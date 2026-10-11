from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import os

from ..models.segmentation  import segment_leaf, detect_disease_regions
from ..models.fuzzy_engine  import FuzzyDiseaseClassifier
from ..models.classifier    import PlantDiseaseClassifier
from ..utils.image_utils    import preprocess_image, generate_overlay, resize_image, image_to_b64
from ..utils.metrics        import calculate_roi_poi

router = APIRouter()

# ── Singletons (loaded once at startup) ──────────────────────────────────────
fuzzy_classifier = FuzzyDiseaseClassifier()

_WEIGHTS = os.path.join(
    os.path.dirname(__file__), '..', '..', 'weights', 'plant_model.pth'
)
disease_classifier = PlantDiseaseClassifier(weights_path=_WEIGHTS)


@router.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Analyse an uploaded plant leaf image and return severity + metrics."""

    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image (jpg/png/webp).')

    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail='Image too large (max 15 MB).')

    # ── Decode ────────────────────────────────────────────────────────────────
    image = preprocess_image(contents)
    if image is None:
        raise HTTPException(status_code=422, detail='Could not decode image file.')

    image = resize_image(image, max_size=640)

    # ── Phase 1: Segmentation ─────────────────────────────────────────────────
    leaf_mask    = segment_leaf(image)
    disease_mask = detect_disease_regions(image, leaf_mask)

    # ── Phase 2: ROI / POI metrics ────────────────────────────────────────────
    roi, poi, roi_normalized = calculate_roi_poi(disease_mask, leaf_mask)

    # ── Phase 3: Disease classification ──────────────────────────────────────
    disease_class, confidence = disease_classifier.predict(image)
    base_recommendation = disease_classifier.get_recommendation(disease_class)

    # ── Phase 4: Fuzzy severity inference ────────────────────────────────────
    severity_score, severity_grade = fuzzy_classifier.classify(poi, roi_normalized)

    # If fuzzy engine or metrics confirm the plant is healthy, reflect it in the disease class
    if severity_grade == 'Healthy' or poi < 5.0:
        disease_class = 'Plant — Healthy'
        confidence = round(max(confidence, severity_score / 100.0), 2)
        base_recommendation = disease_classifier.get_recommendation('healthy')

    # ── Enhance recommendation based on severity (1-2 sentences) ─────────────
    if severity_grade == 'Healthy':
        recommendation = f"{base_recommendation} The plant is in good condition! Continue your current maintenance routine to keep it healthy."
    elif severity_grade == 'Mild':
        recommendation = f"{base_recommendation} The infection is mild. Early treatment now is highly effective and will stop the spread."
    elif severity_grade == 'Medium':
        recommendation = f"{base_recommendation} The plant is in bad condition. Immediate targeted treatment is required. Consider isolating it from healthy plants."
    elif severity_grade == 'Severe':
        recommendation = f"{base_recommendation} The plant is in a critical/deadly state. You may need to heavily prune or entirely remove the plant to protect surrounding crops."
    else:
        recommendation = base_recommendation

    # ── Phase 5: Visual overlay ───────────────────────────────────────────────
    overlay_b64  = generate_overlay(image, leaf_mask, disease_mask)
    original_b64 = image_to_b64(image)

    return JSONResponse(content={
        'severity_grade':        severity_grade,
        'severity_score':        round(severity_score, 2),
        'roi':                   roi,
        'poi':                   round(poi, 2),
        'roi_normalized':        round(roi_normalized, 4),
        'disease_class':         disease_class,
        'confidence':            round(confidence * 100, 1),
        'recommendation':        recommendation,
        'overlay_image':         overlay_b64,
        'original_image':        original_b64,
        'leaf_pixels':           int(np.sum(leaf_mask   > 0)),
        'disease_pixels':        roi,
        'using_fine_tuned_model': disease_classifier.fine_tuned,
    })
