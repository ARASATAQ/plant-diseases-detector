"""
Leaf and disease region segmentation.

Phase 1 (Leaf segmentation):   GrabCut algorithm → leaf vs background mask
Phase 2 (Disease detection):   HSV color analysis → diseased vs healthy tissue

This approach requires no downloaded weights and works on any plant species.
"""
import cv2
import numpy as np


def segment_leaf(image: np.ndarray) -> np.ndarray:
    """
    Separate the leaf from the background using GrabCut.

    Args:
        image: BGR image as ndarray
    Returns:
        Binary mask (255=leaf, 0=background)
    """
    h, w = image.shape[:2]
    mask      = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin = max(5, int(min(h, w) * 0.04))
    rect   = (margin, margin, w - 2 * margin, h - 2 * margin)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        leaf_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception:
        leaf_mask = _green_channel_mask(image)

    # Morphological cleanup: close holes, remove noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k, iterations=2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN,  k, iterations=1)

    # Fallback: if segmented area < 5 % of image, treat whole image as leaf
    if np.sum(leaf_mask > 0) < h * w * 0.05:
        leaf_mask = np.ones((h, w), np.uint8) * 255

    return leaf_mask


def detect_disease_regions(image: np.ndarray,
                            leaf_mask: np.ndarray) -> np.ndarray:
    """
    Identify diseased pixels on the leaf using HSV color analysis.
    Any non-green pixel on the leaf is treated as potentially diseased,
    with additional detection for specific lesion colours.

    Args:
        image     : BGR image
        leaf_mask : binary leaf mask from segment_leaf()
    Returns:
        Binary disease mask (255=diseased, 0=other)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ---- Healthy green tissue ----
    green_mask = cv2.inRange(hsv,
        np.array([25, 30, 30]),
        np.array([90, 255, 255])
    )

    # ---- Disease lesion colours ----
    brown_mask  = cv2.inRange(hsv, np.array([5,  80,  30]), np.array([22, 255, 180]))  # brown spots
    yellow_mask = cv2.inRange(hsv, np.array([22, 80, 150]), np.array([35, 255, 255]))  # chlorosis / yellow
    dark_mask   = cv2.inRange(hsv, np.array([0,   0,   0]), np.array([180, 55,  60]))  # black / dark lesions
    red_mask_1  = cv2.inRange(hsv, np.array([0,  80,  50]), np.array([10, 255, 200]))  # red spots (low H)
    red_mask_2  = cv2.inRange(hsv, np.array([165, 80, 50]), np.array([180, 255, 200])) # red spots (high H)

    # Combine specific disease colours
    disease_colors = cv2.bitwise_or(brown_mask, yellow_mask)
    disease_colors = cv2.bitwise_or(disease_colors, dark_mask)
    disease_colors = cv2.bitwise_or(disease_colors, red_mask_1)
    disease_colors = cv2.bitwise_or(disease_colors, red_mask_2)

    # Non-green leaf pixels (catch any colour anomaly)
    non_green_on_leaf = cv2.bitwise_and(cv2.bitwise_not(green_mask), leaf_mask)

    # Union of both approaches, restricted to leaf area
    disease_mask = cv2.bitwise_or(disease_colors, non_green_on_leaf)
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    # Morphological cleanup: remove tiny speckles, close gaps
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    return disease_mask


def _green_channel_mask(image: np.ndarray) -> np.ndarray:
    """Fallback leaf mask: pixels where G channel dominates R and B."""
    b, g, r = cv2.split(image)
    dominant = (
        (g.astype(int) > r.astype(int) + 10) &
        (g.astype(int) > b.astype(int) + 10)
    )
    mask = dominant.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
