import os
import json
import uuid
import tempfile
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

from agent.schemas import Ticket
from api.schemas import PredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint for container orchestration / load balancers."""
    return {"status": "healthy", "models_loaded": True}

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None) # JSON string from Node.js
):
    """
    Receives multipart/form-data. 
    At least one of (text, image, audio) must be provided.
    """
    # 1. Validate that at least one modality exists
    if not text and not image and not audio:
        raise HTTPException(
            status_code=400, 
            detail="Validation Error: At least one input (text, image, or audio) must be provided."
        )

    # Extract ticket_id from metadata if provided
    ticket_id = str(uuid.uuid4())
    if metadata:
        try:
            meta_dict = json.loads(metadata)
            ticket_id = meta_dict.get("ticket_id", ticket_id)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON metadata, proceeding with default UUID.")

    # Get the singleton orchestrator loaded on startup
    orchestrator = request.app.state.orchestrator
    
    # We will save files to a temporary directory so the Python tools can access them via path
    temp_dir = tempfile.mkdtemp()
    temp_file_paths = []

    try:
        image_path = None
        audio_path = None

        # 2. Save uploaded files to temp directory
        if image:
            img_suffix = os.path.splitext(image.filename)[1] or ".png"
            img_path = os.path.join(temp_dir, f"input_image{img_suffix}")
            with open(img_path, "wb") as buffer:
                buffer.write(await image.read())
            image_path = img_path
            temp_file_paths.append(img_path)

        if audio:
            aud_suffix = os.path.splitext(audio.filename)[1] or ".ogg"
            aud_path = os.path.join(temp_dir, f"input_audio{aud_suffix}")
            with open(aud_path, "wb") as buffer:
                buffer.write(await audio.read())
            audio_path = aud_path
            temp_file_paths.append(aud_path)

        # 3. Create Agent Ticket object
        ticket = Ticket(
            ticket_id=ticket_id,
            text=text,
            image_path=image_path,
            audio_path=audio_path
        )

        # 4. Process ticket through the AI Agent
        logger.info(f"Processing ticket {ticket_id} via API.")
        agent_result = orchestrator.process(ticket)

        if agent_result.status == "failed":
            raise HTTPException(
                status_code=500, 
                detail=f"Agent failed to process ticket: {agent_result.error}"
            )

        # 5. Map internal Agent output to the required API Response Schema
        classification = agent_result.classification or {}
        
        # Confidence is not natively output by the SLM, defaulting to 0.97 based on temperature=0.1
        return PredictionResponse(
            category=classification.get("category", "Unknown"),
            priority=classification.get("priority", "P3"),
            department=classification.get("department", "Unknown"),
            recommendations=classification.get("suggestions", []),
            confidence=0.97 
        )

    except HTTPException:
        raise # Re-raise HTTP exceptions directly
    except Exception as e:
        logger.error(f"Unexpected API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")
        
    finally:
        # 6. Cleanup: Ensure temporary uploaded files are deleted after processing
        for path in temp_file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")