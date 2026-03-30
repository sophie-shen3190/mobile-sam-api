import base64
import io
import os
import urllib.request

import numpy as np
import uvicorn
from fastapi import FastAPI
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

# 启动时加载模型
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
    from mobile_sam import SamPredictor, sam_model_registry
    checkpoint = download_checkpoint()
    model = sam_model_registry["vit_t"](checkpoint=checkpoint)
    model.eval()
    predictor = SamPredictor(model)
    print("MobileSAM loaded.")

class ClickRequest(BaseModel):
    image_base64: str
    point: list  # [x, y] 归一化坐标 0-1

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/click-suggest")
async def click_suggest(req: ClickRequest):
    # 解码图片
    img_data = base64.b64decode(req.image_base64.split(",")[-1])
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
