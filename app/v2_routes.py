from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from typing import Optional
import io
from PIL import Image

# Import the new Pipeline Orchestrator
from app.v2_pipeline import execute_v2_pipeline

v2_router = APIRouter()

@v2_router.post("/analyze")
async def analyze_v2(
    request: Request,
    file: UploadFile = File(...),
    platform_id: Optional[str] = Form("internal_test"),
    caption: Optional[str] = Form(""),
    uploader_id: Optional[str] = Form("user_default")
):
    """
    SENTRY-X V2.0 Core Endpoint
    Runs the 5-phase media integrity firewall architecture.
    """
    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
    metadata = {"caption": caption, "uploader_id": uploader_id}
    
    # Execute the 5-Phase pipeline asynchronously
    result = await execute_v2_pipeline(file_bytes, image, file.filename, platform_id, metadata)
    
    return result
