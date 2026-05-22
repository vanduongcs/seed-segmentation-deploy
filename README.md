---
title: Seed Segmentation Demo
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Seed Segmentation Server Prototype

Minimal deployable server inference core for seed segmentation.

It removes the old app concerns:

- no auth
- no MongoDB
- no history
- no Socket.IO
- no Node backend
- no web/mobile app
- no MobileSAM
- FastSAM-s ONNX is available as an optional quality refiner

Runtime flow:

```text
web browser
-> upload image
-> server YOLOv8 segmentation ONNX
-> optional FastSAM-s ONNX / CPU edge refinement
-> overlay PNG / JSON result
```

This folder is only for the web/server target. The mobile APK target must keep its own TFLite model inside the APK and run inference locally without internet.

## Structure

```text
seed_deploy/
├── app.py
├── requirements.txt
├── Dockerfile
├── grain.settings.json
├── model/
│   └── best.onnx
└── grain_pipeline/
```

## Defaults

The deployment defaults are intentionally conservative:

```text
max_side = 768
conf     = 0.25
iou      = 0.60
max_det  = 300
mode     = balanced
```

These replace the previous heavy prototype defaults such as `conf=0.03` and `max_det=5000`.

Modes:

```text
fast      = YOLO ONNX only, fastest
balanced  = YOLO ONNX + FastSAM only when candidate count is small enough
quality   = YOLO ONNX + FastSAM + CPU mask refine, slowest but best mask quality
```

For dense seed piles, `quality` can be slow because it may refine many individual grain candidates. For customer demos, start with `balanced`; use `quality` for images where mask boundary quality matters more than latency.

## Run Locally With Python

```powershell
cd D:\seed\seed_deploy
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Run With Docker

Build:

```powershell
cd D:\seed\seed_deploy
docker build -t seed-segmentation-prototype .
```

Run:

```powershell
docker run -d --name seed-segmentation-prototype -p 8001:7860 seed-segmentation-prototype
```

Open:

```text
http://127.0.0.1:8001/
```

Stop:

```powershell
docker rm -f seed-segmentation-prototype
```

## API

### GET /health

Returns service status.

### POST /segment

Returns overlay PNG directly.

```powershell
curl.exe -o overlay.png `
  -F "image=@D:\seed\test_images\sample.jpg;type=image/jpeg" `
  "http://127.0.0.1:8001/segment"
```

Optional query parameters:

```text
max_side=768
conf=0.25
iou=0.60
max_det=300
mode=balanced
```

Example:

```powershell
curl.exe -o overlay.png `
  -F "image=@D:\seed\test_images\sample.jpg;type=image/jpeg" `
  "http://127.0.0.1:8001/segment?max_side=768&conf=0.25&max_det=300&mode=balanced"
```

Response headers:

```text
X-Seed-Count
X-Candidate-Count
X-Model
X-Refiner
X-Refiner-Applied
```

### POST /analyze

Returns JSON with summary, measurements, and overlay PNG as base64.

```powershell
curl.exe -o result.json `
  -F "image=@D:\seed\test_images\sample.jpg;type=image/jpeg" `
  "http://127.0.0.1:8001/analyze"
```

Response shape:

```json
{
  "ok": true,
  "image": {
    "width": 700,
    "height": 392,
    "original_width": 700,
    "original_height": 392,
    "scale": 1.0
  },
  "pipeline": "yolo8_nano_segment_deploy",
  "model": "best.onnx",
  "settings": {
    "maxSide": 768,
    "confidence": 0.25,
    "iou": 0.6,
    "maxDet": 300,
    "preprocess": false
  },
  "candidate_count": 297,
  "refiner": "FastSAM-s.onnx",
  "refiner_applied": false,
  "refiner_skip_reason": "candidate_count>80",
  "summary": {
    "count": 286
  },
  "measurements": [],
  "overlay_png_base64": "..."
}
```

## Deployment Notes

Recommended first deployment targets:

- Hugging Face Spaces Docker for a free public AI demo.
- Render Docker Web Service for a simple public API.

Keep the first cloud version CPU-only and YOLO-only. Add storage, auth, history, or mobile inference later.

## Web vs Mobile Architecture

The project has two separate deployment targets:

```text
Web demo
-> model lives on server
-> seed_deploy Docker API runs best.onnx

Mobile APK
-> model lives inside APK
-> Flutter app runs assets/models/best_float16.tflite locally
-> no internet required for normal analysis
```

Do not make the mobile app depend on this Docker service for normal operation. The server API is for web/public demo and optional sync/debug workflows only.
