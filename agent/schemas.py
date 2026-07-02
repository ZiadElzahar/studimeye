from typing import Optional, Dict, Any
from pydantic import BaseModel, model_validator

class Ticket(BaseModel):
    """Input schema for a support ticket."""
    ticket_id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    @model_validator(mode='after')  # ← Use model_validator, NOT field_validator
    def check_at_least_one_modality(self):
        """At least one modality must be provided."""
        if not any([self.text, self.image_path, self.audio_path]):
            raise ValueError("At least one modality (text, image_path, audio_path) must be provided.")
        return self

class AgentResult(BaseModel):
    """Output schema for the agent response."""
    ticket_id: str
    classification: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    status: str
    error: Optional[str] = None