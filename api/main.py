import torch
from fastapi import FastAPI
from api.routes.predict import router

app = FastAPI(
    title="YOLORes18 Inference Service",
    description="REST API for object detection using a custom YOLORes18 architecture and pre-train on VOC 2007 and VOC 2012 datasets",
    version="0.1.0"
)
app.include_router(router)

@app.get("/")
async def root():
    return {
        "service": "YOLORes18 Object Detection API",
        "version": "0.1.0",
        "status": "good",
        "environment": {
            "device_allocated": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "active_threads": torch.get_num_threads()
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }