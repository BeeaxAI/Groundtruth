# GroundTruth — Zero-Hallucination Enterprise Knowledge Agent

> **Gemini Live Agent Challenge 2026** | Category: **Live Agents**

## The Problem

Enterprise AI adoption is blocked by **hallucination**. When AI fabricates answers — citing non-existent policies, inventing contract clauses, or misquoting regulations — the consequences range from costly errors to legal liability. Current chatbots cannot guarantee that every response traces back to verified source documents.

## The Solution

**GroundTruth** is a real-time, multimodal knowledge agent that **never hallucinates**. It uses voice, vision, and text to interact, while a 5-layer grounding architecture ensures every response is cited, validated, and auditable.

### Key Features

| Feature | Description |
|---------|------------|
| **Zero Hallucination** | 5-layer grounding pipeline: strict instructions → BM25 retrieval → citation-enforced prompts → Gemini response → post-validation |
| **Voice Interaction** | Real-time audio via Gemini Live API with barge-in, transcription, natural voice (Kore) |
| **Vision** | Point camera at physical documents — GroundTruth reads and cites them live |
| **Citation Engine** | Every claim tagged with `[Source N]` — hallucinated citations auto-detected |
| **Audit Trail** | Full transparency log: every query, response, citation, and validation |
| **Enterprise Security** | Prompt injection defense, input sanitization, content treated as data not instructions |

## Architecture

```
groundtruth/
├── backend/
│   ├── app.py                     # FastAPI entry point (Phase 8)
│   ├── config.py                  # Pydantic settings (Phase 1)
│   ├── core/
│   │   ├── models.py              # Data models (Phase 2)
│   │   ├── chunker.py             # Paragraph-aware chunking (Phase 2)
│   │   ├── extractor.py           # PDF/DOCX/TXT extraction (Phase 3)
│   │   ├── retriever.py           # BM25 ranking algorithm (Phase 4)
│   │   └── grounding.py           # Zero-hallucination engine (Phase 5-6)
│   ├── api/
│   │   ├── documents.py           # Document CRUD endpoints (Phase 9)
│   │   ├── query.py               # Grounded text query API (Phase 10)
│   │   └── websocket_handler.py   # Live API session manager (Phase 11-14)
│   ├── services/
│   │   └── document_service.py    # Orchestrates extract→chunk→index (Phase 8)
│   ├── utils/
│   │   └── security.py            # Injection defense + rate limiter (Phase 7)
│   └── tests/
│       └── test_core.py           # Unit tests (Phase 23)
├── frontend/
│   ├── index.html                 # App shell (Phase 15)
│   ├── css/styles.css             # Design system (Phase 15)
│   └── js/
│       ├── app.js                 # Main controller (Phase 15-22)
│       ├── audio-engine.js        # Mic recording + playback (Phase 18-19)
│       └── camera-engine.js       # Camera capture @ 1FPS (Phase 20)
├── deployment/
│   ├── deploy.sh                  # One-command Cloud Run deploy
│   └── terraform/main.tf          # Full IaC (bonus points)
├── docs/
│   ├── architecture.mermaid       # System diagram
│   ├── DEMO_SCRIPT.md             # 4-min video script
│   └── SUBMISSION_CHECKLIST.md    # DevPost requirements
├── Dockerfile                     # Multi-stage, non-root
├── run.sh                         # Local dev runner
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | `gemini-2.5-flash-native-audio-preview-12-2025` (Live API) |
| SDK | Google GenAI SDK (`google-genai`) |
| Backend | Python 3.11, FastAPI, WebSockets, Uvicorn |
| Retrieval | BM25 (Okapi) with stopword filtering |
| Document Processing | PyPDF2, python-docx |
| Frontend | Vanilla HTML/JS, Web Audio API, MediaDevices |
| Deployment | Google Cloud Run, Docker, Terraform |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/groundtruth.git
cd groundtruth

# 2. Configure
cp backend/.env.example backend/.env
# Edit backend/.env → add your GOOGLE_API_KEY

# 3. Install & Run
chmod +x run.sh
./run.sh

# Opens at http://localhost:8080
```

### Docker

```bash
docker build -t groundtruth .
docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key groundtruth
```

### Deploy to GCP

```bash
# Shell script
export GCP_PROJECT_ID=your-project GOOGLE_API_KEY=your-key
./deployment/deploy.sh

# Or Terraform
cd deployment/terraform
terraform init && terraform apply
```

## 5-Layer Zero-Hallucination Architecture

1. **Strict System Instructions** — Gemini is told: "Your training data is OFF LIMITS. You only know what the documents say."
2. **BM25 Document Retrieval** — Every query triggers Okapi BM25 ranking over document chunks to find relevant context
3. **Citation-Enforced Prompts** — Context is injected with mandatory `[Source N]` tagging requirements
4. **Gemini Live API Response** — Real-time audio/video processing with grounded context
5. **Post-Response Validation** — Automated checker verifies: citations exist, no phantom sources, refusals are appropriate

## Security

- Prompt injection detection (12 regex patterns)
- Document content treated as DATA, never as INSTRUCTIONS
- Input sanitization (control chars, length limits)
- Rate limiting per client
- Non-root Docker container
- No secrets in code (env-based config)

---

Built for the **Gemini Live Agent Challenge 2026** | [DevPost Submission](https://geminiliveagentchallenge.devpost.com/)
