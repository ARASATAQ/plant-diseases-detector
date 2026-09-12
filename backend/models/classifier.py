"""
Plant disease classifier using ResNet50.
- If fine-tuned weights are present  → returns specific disease class from 38 PlantVillage classes
- If only ImageNet weights available → returns colour-based estimate
RTX 4060 / CUDA auto-detected.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PLANT_CLASSES = [
    'Apple — Apple scab',           'Apple — Black rot',
    'Apple — Cedar apple rust',      'Apple — healthy',
    'Blueberry — healthy',
    'Cherry — Powdery mildew',       'Cherry — healthy',
    'Corn — Cercospora leaf spot',   'Corn — Common rust',
    'Corn — Northern Leaf Blight',   'Corn — healthy',
    'Grape — Black rot',             'Grape — Esca (Black Measles)',
    'Grape — Leaf blight',           'Grape — healthy',
    'Orange — Huanglongbing',
    'Peach — Bacterial spot',        'Peach — healthy',
    'Pepper — Bacterial spot',       'Pepper — healthy',
    'Potato — Early blight',         'Potato — Late blight',
    'Potato — healthy',
    'Raspberry — healthy',
    'Soybean — healthy',
    'Squash — Powdery mildew',
    'Strawberry — Leaf scorch',      'Strawberry — healthy',
    'Tomato — Bacterial spot',       'Tomato — Early blight',
    'Tomato — Late blight',          'Tomato — Leaf Mold',
    'Tomato — Septoria leaf spot',   'Tomato — Spider mites',
    'Tomato — Target Spot',          'Tomato — Yellow Leaf Curl Virus',
    'Tomato — Mosaic virus',         'Tomato — healthy',
]

RECOMMENDATIONS = {
    'healthy':           'Plant looks healthy! Maintain regular watering and fertilization.',
    'Apple scab':        'Apply captan or mancozeb fungicide. Remove and destroy fallen leaves.',
    'Black rot':         'Prune infected wood. Apply copper-based fungicide before rain.',
    'Cedar apple rust':  'Apply myclobutanil fungicide. Remove nearby juniper trees if possible.',
    'Powdery mildew':    'Apply sulfur or potassium bicarbonate. Improve air circulation.',
    'Cercospora':        'Apply chlorothalonil fungicide. Rotate crops next season.',
    'Common rust':       'Apply fungicide (propiconazole). Plant resistant varieties.',
    'Northern Leaf':     'Apply fungicide at early stages. Avoid excessive nitrogen.',
    'Black rot':         'Remove infected clusters. Apply copper fungicide before bloom.',
    'Esca':              'Remove infected wood. No effective chemical cure; manage wounds carefully.',
    'Leaf blight':       'Apply mancozeb or copper fungicide. Remove infected leaves.',
    'Huanglongbing':     'No cure. Remove infected trees. Control psyllid insect vectors.',
    'Bacterial spot':    'Apply copper-based bactericide. Avoid overhead irrigation.',
    'Early blight':      'Apply chlorothalonil or mancozeb. Remove lower infected leaves early.',
    'Late blight':       'Apply systemic fungicide immediately (metalaxyl). Isolate infected plants.',
    'Leaf Mold':         'Improve ventilation. Apply fungicide. Reduce leaf wetness.',
    'Septoria':          'Apply mancozeb fungicide. Stake plants for airflow.',
    'Spider mites':      'Apply miticide or neem oil. Increase humidity. Remove badly infested leaves.',
    'Target Spot':       'Apply fungicide (azoxystrobin). Improve air circulation.',
    'Yellow Leaf Curl':  'Control whitefly vectors. Use reflective mulches. Remove infected plants.',
    'Mosaic virus':      'Remove infected plants. Control aphids. Sanitize tools.',
    'Leaf scorch':       'Check soil pH and moisture. Apply iron chelate if needed.',
    'default':           'Consult an agronomist. Apply appropriate fungicide/bactericide based on visible symptoms.',
}


class PlantDiseaseClassifier:
    def __init__(self, weights_path: str = None):
        self.fine_tuned = False
        # Support both old torchvision (pretrained=True) and new (weights=...)
        try:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(PLANT_CLASSES))


        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=DEVICE)
                # Strip DataParallel 'module.' prefix if present
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                self.fine_tuned = True
                print(f'[Classifier] Loaded fine-tuned weights from {weights_path}')
            except Exception as exc:
                print(f'[Classifier] WARNING: Could not load weights ({exc}). Using ImageNet features.')
        else:
            print('[Classifier] No fine-tuned weights found — using ImageNet ResNet50 + colour analysis.')

        self.model.to(DEVICE).eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bgr: np.ndarray) -> tuple:
        """Returns (disease_class: str, confidence: float 0-1)."""
        if self.fine_tuned:
            return self._model_predict(image_bgr)
        return self._colour_predict(image_bgr)

    def _model_predict(self, image_bgr: np.ndarray) -> tuple:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor  = self.transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)
            conf, idx = torch.max(probs, dim=1)
        return PLANT_CLASSES[idx.item()], round(conf.item(), 4)

    def _colour_predict(self, image_bgr: np.ndarray) -> tuple:
        """Rule-based estimation from lesion colour when model weights absent."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        brown  = int(np.sum((h > 5)  & (h < 22) & (s > 80)))
        yellow = int(np.sum((h > 22) & (h < 35) & (s > 80) & (v > 150)))
        dark   = int(np.sum((v < 60) & (s < 55)))
        total  = brown + yellow + dark + 1
        if dark / total > 0.45:
            return 'Disease — Late blight / Black spot (estimated)', 0.60
        if yellow / total > 0.45:
            return 'Disease — Chlorosis / Mosaic virus (estimated)', 0.60
        if brown / total > 0.35:
            return 'Disease — Early blight / Brown spot (estimated)', 0.60
        return 'Disease — Type undetermined (run download_weights.py for full classifier)', 0.50

    def get_recommendation(self, disease_class: str) -> str:
        lc = disease_class.lower()
        if 'healthy' in lc:
            return RECOMMENDATIONS['healthy']
        for key, rec in RECOMMENDATIONS.items():
            if key.lower() in lc:
                return rec
        return RECOMMENDATIONS['default']
