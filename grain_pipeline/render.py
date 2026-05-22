from __future__ import annotations

import cv2
import numpy as np

from .config import PALETTE


def label_rgb(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. First paint the colored silhouette masks
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        output[labels == label_id] = PALETTE[(int(label_id) - 1) % len(PALETTE)]
        
    # 2. Then draw the actual serial numbers on the centroid of each grain
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        component = (labels == label_id).astype(np.uint8)
        M = cv2.moments(component)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Dynamic font scale based on grain size
            area = M["m00"]
            font_scale = max(0.35, min(0.65, (area / 8000.0) ** 0.5))
            
            text = str(int(label_id))
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = cX - text_size[0] // 2
            text_y = cY + text_size[1] // 2
            
            # Draw black drop shadow outline for high contrast
            cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            # Draw white main text
            cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
    return output


def overlay_rgb(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    colors = label_rgb(labels)
    mask = labels > 0
    output = rgb.copy()
    output[mask] = np.clip((rgb[mask].astype(np.float32) * 0.55) + (colors[mask].astype(np.float32) * 0.45), 0, 255)

    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        component = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, PALETTE[(int(label_id) - 1) % len(PALETTE)], 1)
    return output


def mask_rgb(labels: np.ndarray) -> np.ndarray:
    mask = (labels > 0).astype(np.uint8) * 255
    return np.repeat(mask[..., None], 3, axis=2)


def instance_mask_rgb(instances: list) -> np.ndarray:
    if not instances:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height, width = instances[0].mask.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for instance in instances:
        mask = np.logical_or(mask, instance.mask)
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
