from __future__ import annotations

from pathlib import Path

from .config import PIPELINE_NAME, bool_param, float_param, int_param, model_path
from .fastsam_refine import refine_instances_with_fastsam
from .io import read_image
from .mask_refine import refine_instances_post
from .measure import filter_and_measure, summary_for
from .preprocess import apply_light_preprocessing
from .render import overlay_rgb
from .yolo_segment import predict_instances


def analyze_image(image_path: Path, params: dict | None = None) -> dict:
    params = params or {}

    prepared = read_image(image_path, int_param(params, "maxSide"))
    segment_input = apply_light_preprocessing(prepared.rgb, params)

    instances = predict_instances(segment_input, params)
    yolo_candidate_count = len(instances)
    refiner = "disabled"
    refiner_applied = False
    refiner_skip_reason = ""

    if bool_param(params, "enableSamRefine"):
        candidate_limit = int_param(params, "samCandidateLimit")
        if yolo_candidate_count <= candidate_limit:
            instances = refine_instances_with_fastsam(segment_input, instances, params)
            refiner = "FastSAM-s.onnx"
            refiner_applied = True
        else:
            refiner = "FastSAM-s.onnx"
            refiner_skip_reason = f"candidate_count>{candidate_limit}"

    instances = refine_instances_post(segment_input, instances, params)
    labels, measurements = filter_and_measure(instances, params, prepared.scale)
    overlay = overlay_rgb(segment_input, labels)

    return {
        "image": {
            "width": int(segment_input.shape[1]),
            "height": int(segment_input.shape[0]),
            "original_width": int(prepared.original_width),
            "original_height": int(prepared.original_height),
            "scale": round(float(prepared.scale), 6),
        },
        "pipeline": PIPELINE_NAME,
        "model": model_path(params),
        "settings": {
            "maxSide": int_param(params, "maxSide"),
            "confidence": float_param(params, "yoloConf"),
            "iou": float_param(params, "yoloIou"),
            "maxDet": int_param(params, "yoloMaxDet"),
            "preprocess": bool_param(params, "preprocessImage"),
            "samRefine": bool_param(params, "enableSamRefine"),
            "samCandidateLimit": int_param(params, "samCandidateLimit"),
            "grabCut": bool_param(params, "enableGrabCut"),
            "edgeSnap": bool_param(params, "enableEdgeSnap"),
        },
        "candidate_count": yolo_candidate_count,
        "refined_candidate_count": len(instances),
        "refiner": refiner,
        "refiner_applied": refiner_applied,
        "refiner_skip_reason": refiner_skip_reason,
        "summary": summary_for(measurements),
        "measurements": measurements,
        "overlay": overlay,
    }
