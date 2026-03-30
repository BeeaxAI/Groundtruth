"""
Phase 8: FastAPI application — wires all modules together.
Single entry point for the GroundTruth server.
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, WebSocket, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from utils.security import verify_api_key, get_rate_limiter  # noqa: E402

from config import get_settings  # noqa: E402
from services.document_service import DocumentService  # noqa: E402
from core.grounding import GroundingEngine  # noqa: E402
from api import documents as doc_api  # noqa: E402
from api import query as query_api  # noqa: E402
from api.websocket_handler import LiveSessionHandler  # noqa: E402

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

    logger.info(
        f"GroundTruth server starting on {settings.host}:{settings.port}"
    )
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
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_rate_limiter = get_rate_limiter(
    max_requests=settings.rate_limit_rpm, window_seconds=60
)

# Paths that don't require auth (frontend + OpenAPI docs)
_PUBLIC_PREFIXES = ("/", "/static", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    path = request.url.path

    # Skip auth for public paths
    is_public = any(
        path == p or path.startswith(p + "/")
        for p in _PUBLIC_PREFIXES
    )

    if not is_public:
        # API key check (skipped when api_key is empty — dev mode)
        if settings.api_key:
            provided = request.headers.get("X-API-Key", "")
            if not verify_api_key(provided, settings.api_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )

        # Rate limiting by client IP
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = _rate_limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )

    return await call_next(request)

# Register REST routes
app.include_router(doc_api.router)
app.include_router(query_api.router)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static"
    )


# --- WebSocket ---
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket, api_key: str = ""):
    # Auth check before accepting the connection
    if settings.api_key and not verify_api_key(api_key, settings.api_key):
        await ws.close(code=1008, reason="Invalid or missing API key")
        return

    # Rate limit by client IP before upgrading
    client_ip = ws.client.host if ws.client else "unknown"
    allowed, _ = _rate_limiter.check(client_ip)
    if not allowed:
        await ws.close(code=1008, reason="Rate limit exceeded")
        return

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
