import base64
import io
import os
import traceback
import urllib.request

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None

def download_checkpoint():
    checkpoint_path = "mobile_sam.pt"
    if not os.path.exists(checkpoint_path):
        print("Downloading MobileSAM checkpoint...")
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        print("Download complete.")
    return checkpoint_path

@app.on_event("startup")
async def startup():
    global predictor
    try:
        from mobile_sam import SamPredictor, sam_model_registry
        checkpoint = download_checkpoint()
        model = sam_model_registry["vit_t"](checkpoint=checkpoint)
        model.eval()
        predictor = SamPredictor(model)
        print("MobileSAM loaded successfully.")
    except Exception as e:
        print(f"Failed to load MobileSAM: {e}")
        traceback.print_exc()

class ClickRequest(BaseModel):
    image_base64: str
    point: list  # [x, y] 归一化坐标 0-1

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/click-suggest")
async def click_suggest(req: ClickRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 解码图片
        b64 = req.image_base64.split(",")[-1]
        img_data = base64.b64decode(b64)
        image = np.array(Image.open(io.BytesIO(img_data)).convert("RGB"))
        h, w = image.shape[:2]

        # 归一化坐标转像素
        px = int(req.point[0] * w)
        py = int(req.point[1] * h)

        # SAM 预测
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        # 取得分最高的 mask
        best_mask = masks[np.argmax(scores)]

        # mask 转 bbox
        rows = np.any(best_mask, axis=1)
        cols = np.any(best_mask, axis=0)

        if not rows.any() or not cols.any():
            raise HTTPException(status_code=400, detail="No object detected at this position")

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return {
            "bbox": {
                "x": round(float(x1) / w, 4),
                "y": round(float(y1) / h, 4),
                "width": round(float(x2 - x1) / w, 4),
                "height": round(float(y2 - y1) / h, 4),
            },
            "score": round(float(np.max(scores)), 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
