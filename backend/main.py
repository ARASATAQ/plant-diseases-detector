"""
FastAPI entry point for Plant Disease Detector.
Swagger UI available at http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .routers.predict import router as predict_router

app = FastAPI(
    title='Plant Disease Detector',
    description=(
        'Automated plant disease detection and severity grading using '
        'DeepLabV3+ segmentation + Mamdani Fuzzy Logic inference. '
        'Based on: "Automatic detection and severity analysis of plant disease '
        'based on deep learning and fuzzy logic".'
    ),
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(predict_router, prefix='/api', tags=['Prediction'])

# ── Serve frontend static files ───────────────────────────────────────────────
_FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.isdir(_FRONTEND):
    app.mount('/static', StaticFiles(directory=_FRONTEND), name='static')


@app.get('/', include_in_schema=False)
async def index():
    return FileResponse(os.path.join(_FRONTEND, 'index.html'))


@app.get('/health', tags=['Health'])
async def health():
    import torch
    return {
        'status': 'ok',
        'cuda_available': torch.cuda.is_available(),
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
    }
