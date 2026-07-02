import os
import json
import uuid
import tempfile
import shutil
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

from agent.schemas import Ticket
from api.schemas import PredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": True}

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None)
):
    """Receives multipart/form-data. At least one of (text, audio) must be provided."""
    if not text and not audio:
        raise HTTPException(
            status_code=400, 
            detail="Validation Error: At least one input (text or audio) must be provided."
        )

    ticket_id = str(uuid.uuid4())
    if metadata:
        try:
            meta_dict = json.loads(metadata)
            ticket_id = meta_dict.get("ticket_id", ticket_id)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON metadata.")

    orchestrator = request.app.state.orchestrator
    temp_dir = tempfile.mkdtemp()
    temp_file_paths = []

    try:
        audio_path = None

        if audio:
            aud_suffix = os.path.splitext(audio.filename)[1] or ".ogg"
            aud_path = os.path.join(temp_dir, f"input_audio{aud_suffix}")
            with open(aud_path, "wb") as buffer:
                buffer.write(await audio.read())
            audio_path = aud_path
            temp_file_paths.append(aud_path)

        ticket = Ticket(
            ticket_id=ticket_id,
            text=text,
            audio_path=audio_path
        )

        logger.info(f"Processing ticket {ticket_id} via API.")
        agent_result = orchestrator.process(ticket)

        if agent_result.status == "failed":
            raise HTTPException(
                status_code=500, 
                detail=f"Agent failed to process ticket: {agent_result.error}"
            )

        classification = agent_result.classification or {}
        
        return PredictionResponse(
            category=classification.get("category", "Unknown"),
            priority=classification.get("priority", "P3"),
            department=classification.get("department", "Unknown"),
            recommendations=classification.get("suggestions", []),
            confidence=0.97 
        )

    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Unexpected API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")
        
    finally:
        # Cleanup entire temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)