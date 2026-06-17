from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import io
from torchvision.transforms import functional as f

from api.services.inference import inference_image

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    content = await file.read()

    image_pil = Image.open(io.BytesIO(content))
    image_tensor = f.pil_to_tensor(image_pil)

    prediction = inference_image(image_tensor)

    return prediction