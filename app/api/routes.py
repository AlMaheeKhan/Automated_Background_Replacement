# routes.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
from app.services.segmentation_service import SegmentationService

router = APIRouter()
service = SegmentationService()


@router.post("/replace-background")
async def replace_background(
    image: UploadFile = File(...),
    background: UploadFile = File(None),
):
    image_bytes = await image.read()

    bg_bytes = None
    if background is not None:
        bg_bytes = await background.read()

    result = service.process(image_bytes, bg_bytes)

    return Response(content=result, media_type="image/png")

