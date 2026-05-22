"""CPU-only mask refinement: GrabCut + edge-snap + contour smoothing.

Design principles:
- All operations are per-instance crop (never full-image distance transforms).
- Conservative: if a step makes the mask worse (area collapses or explodes),
  the original mask is kept.
- Fast: O(crop_area) per grain, not O(full_image_area).
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import bool_param, float_param, int_param
from .yolo_segment import InstanceMask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_instances_post(
    rgb: np.ndarray,
    instances: list[InstanceMask],
    params: dict,
) -> list[InstanceMask]:
    """Apply GrabCut → edge-snap → contour smoothing to every instance mask.

    All steps operate on per-instance crops (never the full image), keeping
    the per-grain cost proportional to grain size rather than image size.
    """
    enable_grabcut   = bool_param(params, "enableGrabCut")
    enable_edge_snap = bool_param(params, "enableEdgeSnap")
    smooth_sigma     = float_param(params, "maskContourSmooth")

    if not (enable_grabcut or enable_edge_snap or smooth_sigma > 0):
        return instances

    padding      = int_param(params, "samBoxPadding")
    grabcut_iter = int_param(params, "grabCutIter")
    snap_radius  = int_param(params, "edgeSnapRadius")
    snap_sigma   = float_param(params, "edgeSnapSigma")

    height, width = rgb.shape[:2]
    refined: list[InstanceMask] = []

    for instance in instances:
        original_mask = instance.mask.astype(np.uint8)

        # --- compute crop bbox once, share across steps ---
        ys, xs = np.nonzero(original_mask)
        if len(xs) == 0:
            refined.append(instance)
            continue

        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(width,  int(xs.max()) + 1 + padding)
        y2 = min(height, int(ys.max()) + 1 + padding)

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            refined.append(instance)
            continue

        # Work on a crop mask (uint8 0/1)
        crop_mask = original_mask[y1:y2, x1:x2].copy()
        crop_rgb  = rgb[y1:y2, x1:x2]

        orig_area = int(np.count_nonzero(crop_mask))

        if enable_grabcut:
            crop_mask = _grabcut_refine_crop(crop_rgb, crop_mask, iter_count=grabcut_iter)
            # Safety: if area changed by more than 60%, revert
            new_area = int(np.count_nonzero(crop_mask))
            if orig_area > 0 and (new_area == 0 or new_area / orig_area > 1.6 or new_area / orig_area < 0.4):
                crop_mask = original_mask[y1:y2, x1:x2].copy()

        if enable_edge_snap:
            crop_mask = _edge_snap_crop(crop_rgb, crop_mask,
                                        search_radius=snap_radius, sigma=snap_sigma)
            # Safety check
            new_area = int(np.count_nonzero(crop_mask))
            if orig_area > 0 and (new_area == 0 or new_area / orig_area > 1.5 or new_area / orig_area < 0.3):
                crop_mask = original_mask[y1:y2, x1:x2].copy()

        if smooth_sigma > 0:
            crop_mask = _smooth_contour_crop(crop_mask, sigma=smooth_sigma)

        # Paste refined crop back into a full-size mask
        if not np.any(crop_mask):
            refined.append(instance)
            continue

        full_mask = original_mask.copy()
        full_mask[y1:y2, x1:x2] = crop_mask

        refined.append(
            InstanceMask(
                mask=full_mask.astype(bool),
                confidence=instance.confidence,
                class_id=instance.class_id,
                class_name=instance.class_name,
                source=instance.source,
            )
        )

    return refined


# ---------------------------------------------------------------------------
# GrabCut — operates on crop
# ---------------------------------------------------------------------------

def _grabcut_refine_crop(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    iter_count: int = 3,
) -> np.ndarray:
    """Run GrabCut on a pre-cropped image region."""
    ch, cw = crop_rgb.shape[:2]
    if ch < 4 or cw < 4:
        return crop_mask

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # Build GrabCut mask
    gc_mask = np.full((ch, cw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[crop_mask == 1] = cv2.GC_PR_FGD

    # Erode to get definite foreground core
    k_size = max(3, min(ch, cw) // 8) | 1
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    core    = cv2.erode(crop_mask, kernel, iterations=1)
    gc_mask[core == 1] = cv2.GC_FGD

    if not np.any(gc_mask == cv2.GC_FGD):
        gc_mask[crop_mask == 1] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(crop_bgr, gc_mask, None, bgd_model, fgd_model,
                    iter_count, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return crop_mask

    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    return fg


# ---------------------------------------------------------------------------
# Edge-snap — operates on crop (no full-image distance transforms)
# ---------------------------------------------------------------------------

def _edge_snap_crop(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    search_radius: int = 6,
    sigma: float = 1.5,
) -> np.ndarray:
    """Snap each contour point to the nearest Canny edge within the crop.

    Uses a simple window-scan per contour point (O(pts × r²)) instead of a
    full-image distance-transform-with-labels, keeping per-grain cost small.
    """
    # Compute Canny on this crop only
    gray     = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blurred, threshold1=30, threshold2=80)
    edge_ys, edge_xs = np.nonzero(edges)

    # Use approximated contour to reduce point count
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return crop_mask

    ch, cw = crop_mask.shape[:2]
    result = np.zeros_like(crop_mask)
    new_contours = []

    for contour in contours:
        pts = contour[:, 0, :].astype(np.float64)  # (N, 2) xy

        if sigma > 0:
            pts = _smooth_pts(pts, sigma)

        pts_int = np.clip(np.round(pts).astype(np.int32),
                          [0, 0], [cw - 1, ch - 1])

        if len(edge_xs) == 0:
            # No edges found in crop → keep contour as-is
            new_contours.append(pts_int.reshape(-1, 1, 2))
            continue

        snapped = _snap_pts_to_edges(pts_int, edge_xs, edge_ys, search_radius)
        new_contours.append(snapped.reshape(-1, 1, 2))

    cv2.fillPoly(result, new_contours, 1)
    return result


def _snap_pts_to_edges(
    pts: np.ndarray,       # (N, 2) int [x, y]
    edge_xs: np.ndarray,   # 1-D edge x coords
    edge_ys: np.ndarray,   # 1-D edge y coords
    search_radius: int,
) -> np.ndarray:
    """For each contour point, snap to the nearest edge within search_radius."""
    r2 = search_radius * search_radius
    snapped = pts.copy()

    for i, (x, y) in enumerate(pts):
        dx = edge_xs - x
        dy = edge_ys - y
        d2 = dx * dx + dy * dy
        # Filter to search_radius² window first (fast numpy boolean)
        candidates = np.where(d2 <= r2)[0]
        if len(candidates) == 0:
            continue
        best = candidates[d2[candidates].argmin()]
        snapped[i] = [edge_xs[best], edge_ys[best]]

    return snapped


# ---------------------------------------------------------------------------
# Contour smoothing — operates on crop
# ---------------------------------------------------------------------------

def _smooth_contour_crop(crop_mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Redraw mask from Gaussian-smoothed contour coordinates."""
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return crop_mask

    result = np.zeros_like(crop_mask)
    new_contours = []
    for contour in contours:
        pts = contour[:, 0, :]
        smoothed = _smooth_pts(pts.astype(np.float64), sigma)
        new_contours.append(smoothed.astype(np.int32).reshape(-1, 1, 2))

    cv2.fillPoly(result, new_contours, 1)
    return result


def _smooth_pts(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to a closed polygon (wrap-around padding)."""
    n = len(pts)
    if n < 4:
        return pts
    pad    = min(n, max(3, int(sigma * 3)))
    padded = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0)
    ksize  = int(sigma * 4) * 2 + 1  # must be odd

    sx = cv2.GaussianBlur(padded[:, 0].reshape(-1, 1).astype(np.float32),
                           (1, ksize), sigma).ravel()
    sy = cv2.GaussianBlur(padded[:, 1].reshape(-1, 1).astype(np.float32),
                           (1, ksize), sigma).ravel()
    return np.stack([sx, sy], axis=1)[pad: pad + n]
