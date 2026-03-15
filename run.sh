#!/bin/bash
# GroundTruth — Local Development Runner
# Usage: ./run.sh

set -euo pipefail

echo "=========================================="
echo " GroundTruth — Local Development"
echo "=========================================="

# Check .env
if [ ! -f backend/.env ]; then
    echo "[!] No .env file found. Creating from template..."
    cp backend/.env.example backend/.env
    echo "[!] Edit backend/.env and add your GOOGLE_API_KEY"
    exit 1
fi

# Install deps
echo "[1/3] Installing dependencies..."
pip install -r backend/requirements.txt -q

# Run tests
echo "[2/3] Running tests..."
cd backend
python -c "
import sys; sys.path.insert(0, '.')
from core.chunker import DocumentChunker
from core.retriever import BM25Retriever, tokenize
from core.grounding import GroundingEngine
from core.models import DocumentChunk, Citation, GroundingStatus
from utils.security import InputSanitizer

# Quick smoke tests
assert len(tokenize('hello world')) == 2
chunker = DocumentChunker(100, 20)
assert len(chunker.chunk_document('Test', 'd1', 'f.txt')) == 1
engine = GroundingEngine()
_, _, has = engine.build_grounded_prompt('q', [])
assert not has
print('All smoke tests passed!')
"
cd ..

# Start server
echo "[3/3] Starting server..."
echo ""
echo "  Open: http://localhost:8080"
echo "  Docs: http://localhost:8080/docs"
echo ""
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
