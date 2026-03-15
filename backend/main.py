"""
GroundTruth — Zero-Hallucination Enterprise Knowledge Agent
Main FastAPI server with Gemini Live API integration.

Architecture:
- WebSocket endpoint for real-time audio/video streaming
- REST endpoints for document management
- Gemini Live API for multimodal processing
- Grounding engine ensures every response is cited
"""

import asyncio
import base64
import json
import logging
import os
import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types

from document_store import DocumentStore
from grounding_engine import GroundingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Globals ---
document_store = DocumentStore(chunk_size=500, chunk_overlap=100)
grounding_engine = GroundingEngine(document_store)

# Gemini client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global client
    if GOOGLE_API_KEY:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Gemini client initialized")
    else:
        logger.warning("GOOGLE_API_KEY not set — Gemini features disabled")
    yield
    logger.info("Shutting down GroundTruth server")


app = FastAPI(
    title="GroundTruth",
    description="Zero-Hallucination Enterprise Knowledge Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --- Document Management Endpoints ---

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for grounding. Supports .txt, .pdf, .docx, .md"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content_bytes = await file.read()
    ext = Path(file.filename).suffix.lower()

    try:
        if ext == ".txt" or ext == ".md":
            text_content = content_bytes.decode("utf-8")
        elif ext == ".pdf":
            text_content = _extract_pdf_text(content_bytes)
        elif ext == ".docx":
            text_content = _extract_docx_text(content_bytes)
        else:
            raise HTTPException(
                400, f"Unsupported file type: {ext}. Use .txt, .pdf, .docx, or .md")

        if not text_content.strip():
            raise HTTPException(
                400, "Document appears to be empty or could not be parsed")

        doc = document_store.add_document(file.filename, text_content)
        return {
            "status": "success",
            "doc_id": doc.doc_id,
            "name": doc.name,
            "chunks": len(doc.chunks),
            "content_length": len(doc.content),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    return {"documents": document_store.get_all_documents()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document."""
    if document_store.remove_document(doc_id):
        return {"status": "removed", "doc_id": doc_id}
    raise HTTPException(404, "Document not found")


@app.post("/api/query")
async def text_query(payload: dict):
    """
    Text-based query endpoint (non-streaming fallback).
    For when the user types instead of speaks.
    """
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(400, "Empty query")

    if not client:
        raise HTTPException(
            503, "Gemini client not initialized. Set GOOGLE_API_KEY.")

    # Build grounded prompt
    grounded_prompt, citations, has_context = grounding_engine.build_grounded_prompt(
        query)
    system_instruction = grounding_engine.get_system_instruction()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=grounded_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature for factual grounding
            ),
        )

        response_text = response.text or "I wasn't able to generate a response."
        validation = grounding_engine.validate_response(
            response_text, citations)

        return {
            "response": response_text,
            "citations": citations,
            "validation": validation,
            "has_context": has_context,
        }

    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")


# --- WebSocket for Real-Time Voice/Vision ---

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Real-time WebSocket endpoint for voice + vision interaction.

    Protocol:
    - Client sends JSON messages with type field
    - Types: "audio" (PCM base64), "video" (JPEG base64), "text", "config"
    - Server sends: "audio" (PCM base64), "text" (transcription), "citations", "status"
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    if not client:
        await ws.send_json({"type": "error", "message": "Gemini not configured"})
        await ws.close()
        return

    session = None
    receive_task = None

    try:
        # Configure Live API session
        system_instruction = grounding_engine.get_system_instruction()

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # Professional, clear voice
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        session = await client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        )

        await ws.send_json({"type": "status", "message": "connected"})

        # Task to receive responses from Gemini
        async def receive_from_gemini():
            try:
                accumulated_transcript = ""
                async for response in session.receive():
                    content = response.server_content
                    if content:
                        # Audio response
                        if content.model_turn:
                            for part in content.model_turn.parts:
                                if part.inline_data:
                                    audio_b64 = base64.b64encode(
                                        part.inline_data.data).decode()
                                    await ws.send_json({
                                        "type": "audio",
                                        "data": audio_b64,
                                    })

                        # Output transcription
                        if content.output_transcription and content.output_transcription.text:
                            accumulated_transcript += content.output_transcription.text
                            await ws.send_json({
                                "type": "transcript_output",
                                "text": content.output_transcription.text,
                                "full_text": accumulated_transcript,
                            })

                        # Input transcription (what user said)
                        if content.input_transcription and content.input_transcription.text:
                            user_text = content.input_transcription.text
                            await ws.send_json({
                                "type": "transcript_input",
                                "text": user_text,
                            })

                            # Perform grounding lookup for the transcribed question
                            _, citations, has_context = grounding_engine.build_grounded_prompt(
                                user_text)
                            if citations:
                                await ws.send_json({
                                    "type": "citations",
                                    "citations": citations,
                                    "has_context": has_context,
                                })

                        # Turn complete
                        if content.turn_complete:
                            # Validate the accumulated response
                            if accumulated_transcript:
                                _, citations, _ = grounding_engine.build_grounded_prompt(
                                    accumulated_transcript)
                                validation = grounding_engine.validate_response(
                                    accumulated_transcript, citations)
                                await ws.send_json({
                                    "type": "turn_complete",
                                    "validation": validation,
                                })
                            accumulated_transcript = ""

                        # Interrupted
                        if content.interrupted:
                            await ws.send_json({"type": "interrupted"})
                            accumulated_transcript = ""

            except Exception as e:
                logger.error(f"Gemini receive error: {e}")
                await ws.send_json({"type": "error", "message": str(e)})

        receive_task = asyncio.create_task(receive_from_gemini())

        # Main loop: receive from client, forward to Gemini
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio":
                # Client sends base64 PCM audio
                audio_bytes = base64.b64decode(msg["data"])
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_bytes,
                        mime_type="audio/pcm;rate=16000",
                    )
                )

            elif msg_type == "video":
                # Client sends base64 JPEG frame
                frame_bytes = base64.b64decode(msg["data"])
                await session.send_realtime_input(
                    video=types.Blob(
                        data=frame_bytes,
                        mime_type="image/jpeg",
                    )
                )

            elif msg_type == "text":
                # User types a question — inject grounded context
                user_query = msg.get("text", "")
                grounded_prompt, citations, has_context = grounding_engine.build_grounded_prompt(
                    user_query)

                await session.send_client_content(
                    turns=[{"role": "user", "parts": [
                        {"text": grounded_prompt}]}],
                    turn_complete=True,
                )

                if citations:
                    await ws.send_json({
                        "type": "citations",
                        "citations": citations,
                        "has_context": has_context,
                    })

            elif msg_type == "context_inject":
                # Inject document context into the live session
                # This is called when user asks a question via voice
                # and we want to supplement with document context
                query = msg.get("query", "")
                grounded_prompt, citations, has_context = grounding_engine.build_grounded_prompt(
                    query)

                if has_context:
                    await session.send_client_content(
                        turns=[{
                            "role": "user",
                            "parts": [{"text": f"[DOCUMENT CONTEXT FOR YOUR REFERENCE]\n{grounded_prompt}"}],
                        }],
                        turn_complete=False,
                    )
                    await ws.send_json({
                        "type": "citations",
                        "citations": citations,
                        "has_context": True,
                    })

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if receive_task:
            receive_task.cancel()
        if session:
            try:
                await session.close()
            except Exception:
                pass


# --- Utility Functions ---

def _extract_pdf_text(content_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    from PyPDF2 import PdfReader
    reader = PdfReader(io.BytesIO(content_bytes))
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _extract_docx_text(content_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    from docx import Document
    doc = Document(io.BytesIO(content_bytes))
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


# --- Health & Info ---

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": client is not None,
        "documents_loaded": len(document_store.documents),
        "total_chunks": len(document_store.chunks),
    }


@app.get("/api/audit")
async def audit_log():
    """Return the grounding audit trail for transparency."""
    return {"log": grounding_engine.get_audit_log()}


@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>GroundTruth API</h1><p>Frontend not found. Place index.html in ../frontend/</p>")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
