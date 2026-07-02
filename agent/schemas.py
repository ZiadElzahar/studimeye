from typing import Optional, Dict, Any
from pydantic import BaseModel, validator

class Ticket(BaseModel):
    """Input schema for a support ticket."""
    ticket_id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    @validator('text', 'image_path', 'audio_path', pre=True, always=True)
    def check_at_least_one(cls, v, values, **kwargs):
        # Validation ensures at least one modality is present
        if not any([values.get('text'), v, values.get('audio_path')]):
            raise ValueError("At least one modality (text, image_path, audio_path) must be provided.")
        return v

class AgentResult(BaseModel):
    """Output schema for the agent response."""
    ticket_id: str
    classification: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    status: str
    error: Optional[str] = None