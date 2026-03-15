"""
Phase 8: FastAPI application — wires all modules together.
Single entry point for the GroundTruth server.
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from services.document_service import DocumentService
from core.grounding import GroundingEngine
from api import documents as doc_api
from api import query as query_api
from api.websocket_handler import LiveSessionHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("groundtruth")

# --- Globals ---
settings = get_settings()
doc_service = DocumentService(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    max_documents=settings.max_documents,
)
grounding_engine = GroundingEngine()
gemini_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Gemini client on startup."""
    global gemini_client

    if settings.google_api_key:
        if settings.google_api_key == "dummy_key_for_testing":
            from mock_gemini import MockGeminiClient
            gemini_client = MockGeminiClient(api_key=settings.google_api_key)
            logger.info("Mock Gemini client initialized for testing")
        else:
            try:
                from google import genai
                gemini_client = genai.Client(api_key=settings.google_api_key)
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
    else:
        logger.warning("GOOGLE_API_KEY not set — Gemini features disabled")

    # Initialize Super Memory with Gemini client for embeddings
    doc_service.init_super_memory(
        gemini_client=gemini_client,
        embedding_model=settings.embedding_model,
    )

    # Wire dependencies
    doc_api.init(doc_service)
    query_api.init(doc_service, grounding_engine, gemini_client, settings)

    logger.info(f"GroundTruth server starting on {settings.host}:{settings.port}")
    yield
    logger.info("GroundTruth server shutting down")


# --- App ---
app = FastAPI(
    title="GroundTruth",
    description="Zero-Hallucination Enterprise Knowledge Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST routes
app.include_router(doc_api.router)
app.include_router(query_api.router)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --- WebSocket ---
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    handler = LiveSessionHandler(
        ws=ws,
        gemini_client=gemini_client,
        doc_service=doc_service,
        grounding_engine=grounding_engine,
        settings=settings,
    )
    await handler.run()


# --- Frontend ---
@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse(
        "<h1>GroundTruth API</h1>"
        "<p>Frontend not found. Place index.html in ../frontend/</p>"
        "<p><a href='/docs'>API Documentation</a></p>"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
