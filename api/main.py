import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from agent.orchestrator import Orchestrator
from api.routes import router

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    Executes on startup to load all AI models into memory ONCE (Singleton).
    """
    logger.info("Starting up AI Agent API...")
    logger.info("Loading AI Models into VRAM (this may take a few minutes)...")
    
    try:
        # Initialize the Orchestrator. This loads Phi-3, Qwen, Whisper, and DeepSeek-OCR.
        app.state.orchestrator = Orchestrator(timeout_seconds=300)
        logger.info("AI Orchestrator initialized successfully. API is ready for traffic.")
    except Exception as e:
        logger.critical(f"Fatal error during model startup: {e}", exc_info=True)
        # Prevent startup if models fail to load
        raise e
        
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down AI Agent API. Unloading models.")
    app.state.orchestrator = None

# Initialize FastAPI App
app = FastAPI(
    title="AI Support Agent API",
    description="API for routing and processing multimodal customer support tickets.",
    version="1.0.0",
    lifespan=lifespan
)

# Include API Routes
app.include_router(router)

# ==========================================
# Global Exception Handlers
# ==========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic/Form validation errors gracefully."""
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Input validation failed.",
            "errors": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions to prevent stack traces leaking to Node.js."""
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal server error occurred.",
            "error": str(exc)
        }
    )