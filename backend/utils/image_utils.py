"""
Image preprocessing and overlay generation utilities.
"""
import cv2
import numpy as np
import base64


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to OpenCV BGR ndarray."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def resize_image(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """Resize maintaining aspect ratio so the longest side <= max_size."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def generate_overlay(original: np.ndarray,
                     leaf_mask: np.ndarray,
                     disease_mask: np.ndarray) -> str:
    """
    Render a color overlay:
      - Semi-transparent green on healthy leaf tissue
      - Semi-transparent red on diseased regions
    Returns base64-encoded PNG data-URI.
    """
    overlay = original.copy().astype(np.float32)

    # Healthy leaf (leaf minus disease)
    healthy_mask = cv2.bitwise_and(
        leaf_mask, cv2.bitwise_not(disease_mask)
    )

    # Green tint on healthy tissue
    green_layer = np.zeros_like(overlay)
    green_layer[:, :] = [0, 180, 0]
    alpha_h = (healthy_mask[:, :, np.newaxis] / 255.0) * 0.30
    overlay = overlay * (1 - alpha_h) + green_layer * alpha_h

    # Red tint on diseased regions
    red_layer = np.zeros_like(overlay)
    red_layer[:, :] = [0, 0, 220]
    alpha_d = (disease_mask[:, :, np.newaxis] / 255.0) * 0.55
    overlay = overlay * (1 - alpha_d) + red_layer * alpha_d

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw contours of disease regions
    contours, _ = cv2.findContours(
        disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    _, buf = cv2.imencode('.png', overlay)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def image_to_b64(image: np.ndarray) -> str:
    """Encode any OpenCV image to base64 data-URI PNG."""
    _, buf = cv2.imencode('.png', image)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/png;base64,{b64}'
