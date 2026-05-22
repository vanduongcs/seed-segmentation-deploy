from __future__ import annotations

import tempfile
import base64
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response

from grain_pipeline.pipeline import analyze_image

app = FastAPI(title="Seed Segmentation Prototype", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Seed Segmentation Prototype</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }
    main { max-width: 980px; margin: 0 auto; }
    h1 { margin-bottom: 8px; }
    form { display: grid; gap: 16px; padding: 20px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }
    label { display: grid; gap: 6px; font-weight: 700; }
    input { padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; }
    .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    button { width: max-content; padding: 11px 18px; border: 0; border-radius: 6px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    #status { margin: 18px 0; font-weight: 700; }
    img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; background: white; }
    @media (max-width: 720px) { .row { grid-template-columns: 1fr; } body { margin: 16px; } }
  </style>
</head>
<body>
  <main>
    <h1>Seed Segmentation Server Prototype</h1>
    <p>Web/server demo: chọn ảnh JPG/PNG, bấm Analyze. Balanced dùng YOLO ONNX và tự bật FastSAM khi ảnh không quá dày; Quality ưu tiên mask đẹp hơn nhưng chậm hơn.</p>

    <form id="form">
      <label>
        Ảnh hạt giống
        <input id="image" name="image" type="file" accept="image/png,image/jpeg" required />
      </label>
      <div class="row">
        <label>mode
          <select name="mode">
            <option value="balanced" selected>balanced</option>
            <option value="fast">fast</option>
            <option value="quality">quality</option>
          </select>
        </label>
        <label>max_side <input name="max_side" type="number" min="128" max="1024" value="768" /></label>
        <label>conf <input name="conf" type="number" min="0.05" max="0.99" step="0.01" value="0.25" /></label>
        <label>max_det <input name="max_det" type="number" min="1" max="300" value="300" /></label>
      </div>
      <button id="button" type="submit">Analyze</button>
    </form>

    <div id="status"></div>
    <img id="result" alt="" />
  </main>

  <script>
    const form = document.getElementById("form");
    const button = document.getElementById("button");
    const statusBox = document.getElementById("status");
    const result = document.getElementById("result");
    let currentUrl = "";

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      result.removeAttribute("src");
      statusBox.textContent = "Đang xử lý...";
      button.disabled = true;

      const data = new FormData(form);
      const params = new URLSearchParams({
        max_side: data.get("max_side"),
        conf: data.get("conf"),
        max_det: data.get("max_det"),
        mode: data.get("mode"),
      });
      const upload = new FormData();
      upload.append("image", data.get("image"));

      const started = performance.now();
      try {
        const response = await fetch(`/segment?${params}`, { method: "POST", body: upload });
        if (!response.ok) throw new Error(await response.text());
        const blob = await response.blob();
        currentUrl = URL.createObjectURL(blob);
        result.src = currentUrl;
        const seconds = ((performance.now() - started) / 1000).toFixed(2);
        statusBox.textContent = `Xong: ${seconds}s | count=${response.headers.get("x-seed-count")} | candidates=${response.headers.get("x-candidate-count")}`;
      } catch (error) {
        statusBox.textContent = `Lỗi: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "seed-segmentation-prototype"}


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    max_side: int = Query(768, ge=128, le=1024),
    conf: float = Query(0.25, ge=0.05, le=0.99),
    iou: float = Query(0.60, ge=0.05, le=0.95),
    max_det: int = Query(300, ge=1, le=300),
    mode: Literal["fast", "balanced", "quality"] = Query("balanced"),
) -> Response:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")

    suffix = ".jpg" if image.content_type == "image/jpeg" else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(await image.read())

    try:
        result = analyze_image(temp_path, _params_for_mode(max_side, conf, iou, max_det, mode))
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode overlay")
        headers = {
            "X-Seed-Count": str(result["summary"]["count"]),
            "X-Candidate-Count": str(result["candidate_count"]),
            "X-Refiner": str(result["refiner"]),
            "X-Refiner-Applied": str(result["refiner_applied"]).lower(),
            "X-Model": Path(result["model"]).name,
        }
        return Response(content=buffer.tobytes(), media_type="image/png", headers=headers)
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    max_side: int = Query(768, ge=128, le=1024),
    conf: float = Query(0.25, ge=0.05, le=0.99),
    iou: float = Query(0.60, ge=0.05, le=0.95),
    max_det: int = Query(300, ge=1, le=300),
    mode: Literal["fast", "balanced", "quality"] = Query("balanced"),
) -> dict:
    result, overlay_png = await _run_analysis(image, max_side, conf, iou, max_det, mode)
    return {
        "ok": True,
        "image": result["image"],
        "pipeline": result["pipeline"],
        "model": Path(result["model"]).name,
        "settings": result["settings"],
        "candidate_count": result["candidate_count"],
        "refined_candidate_count": result["refined_candidate_count"],
        "refiner": result["refiner"],
        "refiner_applied": result["refiner_applied"],
        "refiner_skip_reason": result["refiner_skip_reason"],
        "summary": result["summary"],
        "measurements": result["measurements"],
        "overlay_png_base64": base64.b64encode(overlay_png).decode("ascii"),
    }


async def _run_analysis(
    image: UploadFile,
    max_side: int,
    conf: float,
    iou: float,
    max_det: int,
    mode: Literal["fast", "balanced", "quality"],
) -> tuple[dict, bytes]:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")

    suffix = ".jpg" if image.content_type == "image/jpeg" else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(await image.read())

    try:
        result = analyze_image(temp_path, _params_for_mode(max_side, conf, iou, max_det, mode))
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode overlay")
        result = {key: value for key, value in result.items() if key != "overlay"}
        return result, buffer.tobytes()
    finally:
        temp_path.unlink(missing_ok=True)


def _params_for_mode(
    max_side: int,
    conf: float,
    iou: float,
    max_det: int,
    mode: Literal["fast", "balanced", "quality"],
) -> dict:
    params = {
        "maxSide": max_side,
        "yoloConf": conf,
        "yoloIou": iou,
        "yoloMaxDet": max_det,
        "enableSamRefine": False,
        "samCandidateLimit": 80,
        "enableGrabCut": False,
        "enableEdgeSnap": False,
        "maskContourSmooth": 0.0,
    }
    if mode == "balanced":
        params.update({
            "enableSamRefine": True,
            "samCandidateLimit": 80,
            "maskContourSmooth": 0.8,
        })
    elif mode == "quality":
        params.update({
            "enableSamRefine": True,
            "samCandidateLimit": max_det,
            "enableGrabCut": True,
            "enableEdgeSnap": True,
            "maskContourSmooth": 1.0,
        })
    return params
