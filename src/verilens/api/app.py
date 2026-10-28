"""
FastAPI application for VeriLens AI.
Provides /health and /analyze endpoints with strict validation, upload limits, and CORS.
"""

from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from verilens.core.config import settings
from verilens.core.logger import get_logger
from verilens.detectors.registry import registry
from verilens.pipeline.orchestrator import pipeline
from verilens.schemas.response import (
    HealthResponse,
    LimitsMeta,
    AnalysisResponse,
    AnalysisRequest,
)

logger = get_logger("API")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Misinformation Detection & Explanation Engine",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check and capability advertisement endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        limits=LimitsMeta(
            max_media_size_bytes=settings.MAX_MEDIA_SIZE_BYTES,
            max_media_size_mb=settings.MAX_MEDIA_SIZE_BYTES / (1024 * 1024),
            max_text_chars=settings.MAX_TEXT_CHARS,
            allowed_media_types=sorted(list(settings.ALLOWED_MEDIA_TYPES)),
        ),
        detectors=registry.list_metadata(),
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    text: Optional[str] = Form(None),
    media_file: Optional[UploadFile] = File(None),
    enabled_detectors: Optional[str] = Form(None),
):
    """
    Multimodal misinformation analysis endpoint.
    Accepts multipart/form-data for file uploads, or JSON for text-only claims.
    """
    media_bytes: Optional[bytes] = None
    media_filename: Optional[str] = None
    media_content_type: Optional[str] = None
    active_detectors: Optional[List[str]] = None

    # Handle application/json requests
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            req_model = AnalysisRequest(**body)
            text = req_model.text
            active_detectors = req_model.enabled_detectors
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON payload: {str(exc)}",
            )
    else:
        if enabled_detectors:
            active_detectors = [d.strip() for d in enabled_detectors.split(",") if d.strip()]

    # Validate inputs
    if not text and not media_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one input modality ('text' or 'media_file') must be provided.",
        )

    if text and len(text) > settings.MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Text length exceeds maximum allowed limit ({settings.MAX_TEXT_CHARS} chars).",
        )

    # Validate media file if present
    if media_file is not None:
        media_content_type = media_file.content_type
        media_filename = media_file.filename

        # MIME type whitelist verification
        if media_content_type and media_content_type.lower() not in settings.ALLOWED_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported media type '{media_content_type}'. "
                    f"Allowed types: {', '.join(sorted(list(settings.ALLOWED_MEDIA_TYPES)))}"
                ),
            )

        # Read file with size enforcement
        media_bytes = await media_file.read()
        if len(media_bytes) > settings.MAX_MEDIA_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Uploaded media ({len(media_bytes)} bytes) exceeds the {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)}MB limit."
                ),
            )

    try:
        response = pipeline.run(
            text=text,
            media_bytes=media_bytes,
            media_filename=media_filename,
            media_content_type=media_content_type,
            enabled_detectors=active_detectors,
        )
        return response
    except Exception as exc:
        logger.exception(f"Unhandled error in analysis pipeline: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(exc)}",
        )
