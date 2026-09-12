"""
ROI and POI metric calculation for plant disease quantification.
ROI (Region of Interest)  = absolute count of diseased pixels
POI (Percentage of Infection) = diseased pixels / leaf pixels * 100
"""
import numpy as np


def calculate_roi_poi(disease_mask: np.ndarray, leaf_mask: np.ndarray) -> tuple:
    """
    Calculate ROI and POI from segmentation masks.

    Args:
        disease_mask : Binary mask, 255 = diseased pixel, 0 = other
        leaf_mask    : Binary mask, 255 = leaf pixel, 0 = background

    Returns:
        (roi, poi, roi_normalized)
        roi            : int   — absolute diseased pixel count
        poi            : float — percentage of leaf that is diseased (0–100)
        roi_normalized : float — roi / total image pixels (0–1)
    """
    total_pixels  = disease_mask.shape[0] * disease_mask.shape[1]
    leaf_pixels   = int(np.sum(leaf_mask   > 0))
    disease_pixels = int(np.sum(disease_mask > 0))

    roi = disease_pixels
    poi = (disease_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0.0
    roi_normalized = disease_pixels / total_pixels if total_pixels > 0 else 0.0

    return roi, round(poi, 2), round(roi_normalized, 4)
