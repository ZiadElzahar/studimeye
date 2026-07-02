from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    """The standardized JSON response returned to the Node.js backend."""
    category: str
    priority: str
    department: str
    recommendations: List[str]
    confidence: float