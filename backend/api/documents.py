"""
Phase 9: REST API — Document management endpoints.
Upload, list, delete documents.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Injected at app startup
_doc_service = None


def init(doc_service):
    global _doc_service
    _doc_service = doc_service


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for grounding. Supports PDF, DOCX, TXT, MD."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content_bytes = await file.read()

    if len(content_bytes) == 0:
        raise HTTPException(400, "File is empty")

    max_size = 20 * 1024 * 1024  # 20MB
    if len(content_bytes) > max_size:
        size_mb = len(content_bytes) / 1024 / 1024
        raise HTTPException(
            400, f"File too large ({size_mb:.1f}MB). Maximum: 20MB"
        )

    try:
        result = _doc_service.ingest(file.filename, content_bytes)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ImportError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@router.get("")
async def list_documents():
    """List all uploaded documents."""
    return {"documents": _doc_service.get_all_documents()}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and its indexed chunks."""
    if _doc_service.remove(doc_id):
        return {"status": "removed", "doc_id": doc_id}
    raise HTTPException(404, "Document not found")


@router.get("/health")
async def all_document_health():
    """Get health scores for all documents — richness, diversity, embedding coverage."""
    return {"scores": _doc_service.get_document_health()}


@router.get("/memory/stats")
async def memory_stats():
    """Get Super Memory compression statistics."""
    return _doc_service.get_memory_stats()


@router.get("/memory/duplicates")
async def memory_duplicates():
    """Detect near-duplicate documents using SimHash fingerprinting."""
    return {"duplicates": _doc_service.find_duplicates()}


@router.get("/gaps")
async def knowledge_gaps():
    """Get knowledge gap analysis — topics users asked about that couldn't be answered well."""
    return _doc_service.get_knowledge_gaps()


@router.get("/heatmap")
async def all_heatmaps():
    """Get hallucination heatmap for all documents — per-chunk citation frequency."""
    return {"heatmaps": _doc_service.get_heatmap()}


@router.get("/{doc_id}/heatmap")
async def single_document_heatmap(doc_id: str):
    """Get hallucination heatmap for a single document."""
    heatmap = _doc_service.get_heatmap(doc_id)
    if not heatmap:
        raise HTTPException(404, "Document not found")
    return heatmap


@router.get("/{doc_id}/health")
async def single_document_health(doc_id: str):
    """Get health score for a single document."""
    health = _doc_service.get_document_health(doc_id)
    if not health:
        raise HTTPException(404, "Document not found")
    return health


@router.get("/{doc_id}/insights")
async def document_insights(doc_id: str):
    """Get AI-extracted insights (keywords, summary) for a document."""
    insights = _doc_service.get_document_insights(doc_id)
    if not insights:
        raise HTTPException(
            404, "Document not found or Super Memory not initialized")
    return insights
