from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class GenerateRequest(BaseModel):
    """
    Request model for image generation.
    """
    seed: Optional[int] = Field(None, description="Random seed for generation reproducibility.")
    truncation_psi: float = Field(0.7, ge=0.0, le=2.0, description="Truncation psi for style mixing (0.0 to 2.0).")
    
class GenerateResponse(BaseModel):
    """
    Response model for image generation.
    """
    image: str = Field(..., description="Base64 encoded PNG image.")
    meta: Dict[str, Any] = Field(..., description="Metadata including seed, latency, etc.")

class HealthCheckResponse(BaseModel):
    """
    Response model for health check.
    """
    status: str
    version: str
