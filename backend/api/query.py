"""
Phase 10: REST API — Grounded text query endpoint.
Fallback for when WebSocket/voice isn't used.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])

# Injected at app startup
_doc_service = None
_grounding_engine = None
_gemini_client = None
_settings = None


def init(doc_service, grounding_engine, gemini_client, settings):
    global _doc_service, _grounding_engine, _gemini_client, _settings
    _doc_service = doc_service
    _grounding_engine = grounding_engine
    _gemini_client = gemini_client
    _settings = settings


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)


@router.post("/query")
async def text_query(request: QueryRequest):
    """Text-based grounded query (non-streaming REST fallback)."""
    if not _gemini_client:
        raise HTTPException(503, "Gemini not configured. Set GOOGLE_API_KEY.")

    from utils.security import InputSanitizer
    sanitizer = InputSanitizer()
    clean_query, warnings = sanitizer.sanitize_query(request.query)

    if not clean_query:
        raise HTTPException(400, "Query is empty after sanitization")

    # Retrieve relevant chunks (hybrid: BM25 + semantic + hierarchical)
    relevant = await _doc_service.search_hybrid(clean_query, top_k=_settings.max_retrieval_chunks)

    # Build grounded prompt
    grounded_prompt, citations, has_context = _grounding_engine.build_grounded_prompt(
        clean_query, relevant, max_context_chars=_settings.max_context_chars,
        has_documents=_doc_service.has_documents(),
    )

    system_instruction = _grounding_engine.build_system_instruction(
        _doc_service.get_document_summary()
    )

    try:
        from google.genai import types

        response = _gemini_client.models.generate_content(
            model=_settings.gemini_text_model,
            contents=grounded_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=_settings.gemini_temperature,
                max_output_tokens=_settings.gemini_max_output_tokens,
            ),
        )

        response_text = response.text or "I wasn't able to generate a response."

        # Validate grounding
        validation = _grounding_engine.validate_response(response_text, citations)

        # Audit log
        _grounding_engine.log_query(
            clean_query, citations, has_context, validation, response_text
        )

        result = {
            "response": response_text,
            "citations": [c.to_dict() for c in citations],
            "validation": validation.to_dict(),
            "has_context": has_context,
        }

        if warnings:
            result["security_warnings"] = warnings

        return result

    except Exception as e:
        logger.error(f"Gemini query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query failed: {str(e)}")


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": _gemini_client is not None,
        "documents_loaded": _doc_service.document_count,
        "total_chunks": _doc_service.total_chunks,
    }


@router.get("/audit")
async def audit_log():
    """Return the grounding audit trail."""
    return {"log": _grounding_engine.get_audit_log()}
