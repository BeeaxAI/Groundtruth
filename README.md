# GroundTruth — Zero-Hallucination Enterprise Knowledge Agent

> **Gemini Live Agent Challenge 2026** | Category: **Live Agents**

## The Problem

Enterprise AI adoption is blocked by **hallucination**. When AI fabricates answers — citing non-existent policies, inventing contract clauses, or misquoting regulations — the consequences range from costly errors to legal liability. Current chatbots cannot guarantee that every response traces back to verified source documents.

## The Solution

**GroundTruth** is a real-time, multimodal knowledge agent that **never hallucinates**. It uses voice, vision, and text to interact, while a 5-layer grounding architecture ensures every response is cited, validated, and auditable.

### Core Features

| Feature | Description |
|---------|------------|
| **Zero Hallucination** | 5-layer grounding pipeline: strict instructions → retrieval → citation-enforced prompts → Gemini response → post-validation |
| **Super Memory** | 5 mathematical compression techniques for intelligent document storage and hybrid semantic search |
| **Voice Interaction** | Real-time audio via Gemini Live API with barge-in, transcription, natural voice (Kore) |
| **Vision** | Point camera at physical documents — GroundTruth reads and cites them live |
| **Citation Engine** | Every claim tagged with `[Source N]` — hallucinated citations auto-detected |
| **Audit Trail** | Full transparency log: every query, response, citation, and validation |
| **Enterprise Security** | Prompt injection defense, input sanitization, content treated as data not instructions |

### Intelligence Features (10 Advanced Capabilities)

| Feature | Description |
|---------|------------|
| **Confidence Score Meter** | Real-time radial gauge per response with formula: `C = 0.4×Coverage + 0.3×Sources + 0.3×Grounding`. Includes info button with interactive breakdown tooltip |
| **Smart Follow-up Chips** | 3-strategy algorithm generates contextual follow-up questions from adjacent chunks, Super Memory keywords, and bigram analysis |
| **Document Health Score** | Per-document quality grade (Excellent/Good/Fair/Low) computed from content richness, keyword diversity, embedding coverage, and structure quality |
| **Hallucination Heatmap** | Visual citation density strip per document — shows which chunks get cited most (frozen → cold → warm → hot color scale) |
| **Knowledge Gap Detector** | Tracks low-confidence queries, extracts bigram/trigram phrases, clusters by frequency, and suggests specific documents to upload |
| **Session Analytics** | Real-time dashboard tracking total queries, average confidence, grounded rate, session duration, and confidence range |
| **Answer Export** | One-click download of full conversation with citations, grounding status, confidence scores, and session statistics as formatted .txt |
| **Auto-Tag Documents** | Extracts keywords from Super Memory insights and displays them as colored tag chips on each uploaded document |
| **Citation Deep-Links** | Click any `[Source N]` in chat to auto-scroll and highlight the matching citation card in the right panel |
| **Session Keep-Alive** | Silent 10ms PCM audio frames sent every 15 seconds to prevent Gemini Live API session timeouts |

### UI/UX Features

| Feature | Description |
|---------|------------|
| **Premium Dark/Light Theme** | Toggle between dark mode and a color-graded white theme with warm ivory tones, glass morphism, layered card depth, and blue-tinted shadows. Persists via localStorage |
| **Collapsible Side Panels** | Both sidebar and right panel collapse/expand with smooth 0.4s animations, floating re-expand tabs, and persistent state across sessions |

## Architecture

```
groundtruth/
├── backend/
│   ├── app.py                     # FastAPI entry point with lifespan & Super Memory init
│   ├── config.py                  # Pydantic settings (gemini models, chunking, audio, security)
│   ├── core/
│   │   ├── models.py              # Data models (Document, Chunk, Citation, GroundingResult)
│   │   ├── chunker.py             # Paragraph-aware chunking with overlap
│   │   ├── extractor.py           # PDF/DOCX/TXT/MD extraction (PyPDF2, python-docx)
│   │   ├── retriever.py           # BM25 Okapi ranking algorithm
│   │   ├── grounding.py           # Zero-hallucination engine + confidence scoring
│   │   └── super_memory.py        # 5-technique compression engine (853 lines)
│   ├── api/
│   │   ├── documents.py           # Document CRUD + health + heatmap + gaps + insights
│   │   ├── query.py               # Grounded text query API (REST fallback)
│   │   └── websocket_handler.py   # Live API session manager with keep-alive
│   ├── services/
│   │   └── document_service.py    # Orchestrates extract→chunk→index→memory + analytics
│   ├── utils/
│   │   └── security.py            # Injection defense + rate limiter
│   └── tests/
│       └── test_core.py           # Unit tests
├── frontend/
│   ├── index.html                 # App shell with theme toggle & collapsible panels
│   ├── css/styles.css             # Design system (dark + premium white theme, ~1400 lines)
│   └── js/
│       ├── app.js                 # Main controller (~970 lines, 12 feature modules)
│       ├── audio-engine.js        # Mic recording + playback (Web Audio API)
│       └── camera-engine.js       # Camera capture @ 1FPS (MediaDevices)
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
| AI Model (Live) | `gemini-2.5-flash-native-audio-preview-12-2025` (real-time audio/video) |
| AI Model (Text) | `gemini-2.5-flash-preview-05-20` (REST fallback) |
| Embeddings | `gemini-embedding-001` (3072-dim → 96 bytes binary quantized) |
| SDK | Google GenAI SDK (`google-genai`) |
| Backend | Python 3.11, FastAPI, WebSockets, Uvicorn |
| Retrieval | Hybrid: BM25 (Okapi) + Binary Embeddings + Hierarchical Routing + RRF |
| Document Processing | PyPDF2, python-docx |
| Math/Compression | NumPy (binary quantization, Hamming distance) |
| Frontend | Vanilla HTML/JS/CSS, Web Audio API, MediaDevices |
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
2. **Hybrid Document Retrieval** — BM25 keyword matching + binary embedding semantic search + hierarchical routing via Reciprocal Rank Fusion
3. **Citation-Enforced Prompts** — Context is injected with mandatory `[Source N]` tagging requirements
4. **Gemini Live API Response** — Real-time audio/video processing with grounded context
5. **Post-Response Validation** — Automated checker verifies: citations exist, no phantom sources, refusals are appropriate

## Super Memory — Compressed Knowledge Engine

GroundTruth includes a **Super Memory** system that stores document knowledge in highly compressed form using 5 mathematical techniques. This enables faster retrieval, semantic search, and efficient memory usage.

### Technique 1: Binary Quantized Embeddings (32x compression)

Converts Gemini `gemini-embedding-001` float vectors into compact binary representations.

```
Math:  b_i = 1 if e_i > 0, else 0
Size:  3,072 × float32 = 12,288 bytes → 3,072 bits = 384 bytes (32x compression)
Speed: Hamming distance via XOR + popcount (~100x faster than cosine)
```

**Why it works:** The sign of each embedding dimension preserves angular relationships between vectors. For two vectors with angle θ: `P(sign(a_i) = sign(b_i)) = 1 - θ/π`, so Hamming similarity approximates cosine similarity.

### Technique 2: Bloom Filter Topic Index (~15-40x compression)

Probabilistic data structure for instant "does this document mention topic X?" checks.

```
Math:  m = -n × ln(p) / (ln 2)²    (optimal bit array size)
       k = (m/n) × ln 2             (optimal hash functions)
       FPR = (1 - e^(-kn/m))^k      (false positive rate)
Space: 1000 keywords → 1.2 KB (vs ~15-50 KB raw strings)
Speed: O(1) lookup with zero false negatives
```

### Technique 3: SimHash Fingerprinting (12,500x compression)

Locality-Sensitive Hashing for near-duplicate document detection.

```
Algorithm:
  1. Tokenize document → weighted features {w_i: tf(w_i)}
  2. For each feature: hash to 64 bits, vote on each position
  3. Fingerprint = sign of accumulated votes

Size:  100KB document → 8-byte fingerprint
Match: hamming(h1, h2) ≤ 3 → documents are >95% similar
```

### Technique 4: Hierarchical Memory Levels

Multi-resolution knowledge storage with smart query routing:

```
Level 3 (Bloom):    ~1.2 KB/doc  → "Does doc mention X?"     O(1)
Level 2 (Keywords): ~2.5 KB/doc  → Top-50 TF-IDF terms       O(k)
Level 1 (Summary):  ~1 KB/doc    → Top-5 key sentences        O(1)
Level 0 (Chunks):   Full size    → Raw text for deep search   O(n)

Query routing eliminates ~80% of documents before expensive search.
```

### Technique 5: Reciprocal Rank Fusion (Hybrid Search)

Combines 3 independent retrieval signals into a single optimal ranking.

```
Math:  RRF(d) = Σ  1 / (k + rank_i(d))    where k = 60
                i

Signals:
  1. BM25 keyword ranking     → catches exact word matches
  2. Binary embedding search  → catches semantic/paraphrase matches
  3. Hierarchical routing     → eliminates off-topic documents

Example:
  Query: "What are the project deadlines?"
  BM25 finds:      chunks with word "deadline"
  Semantic finds:  chunks about "due dates", "timeline", "schedule"
  Hierarchy skips: financial documents, HR policies
  Fusion:          merges all signals for best overall ranking
```

### Memory Budget

For 100 documents × 50 chunks each:

```
Binary embeddings: 5,000 × 96 bytes  =  480 KB
Bloom filters:     100 × 1.2 KB      =  120 KB
Keywords:          100 × 2.5 KB      =  250 KB
Summaries:         100 × 1 KB        =  100 KB
SimHash:           100 × 8 bytes     =  0.8 KB
──────────────────────────────────────────────
Total compressed:                    ≈  951 KB
Raw text storage:  100 × 50 KB       = 5,000 KB
Compression ratio:                     ~5.3x
```

### API Endpoints

| Endpoint | Description |
|----------|------------|
| `POST /api/documents/upload` | Upload PDF, DOCX, TXT, or MD files |
| `GET /api/documents` | List all documents with metadata |
| `DELETE /api/documents/{doc_id}` | Remove a document |
| `GET /api/documents/{doc_id}/health` | Document health score with breakdown |
| `GET /api/documents/health` | Health scores for all documents |
| `GET /api/documents/{doc_id}/heatmap` | Citation heatmap per chunk |
| `GET /api/documents/heatmap` | Heatmaps for all documents |
| `GET /api/documents/gaps` | Knowledge gap analysis with upload suggestions |
| `GET /api/documents/{doc_id}/insights` | Auto-tags, keywords, summary per document |
| `GET /api/documents/memory/stats` | Super Memory compression statistics |
| `GET /api/documents/memory/duplicates` | Near-duplicate detection via SimHash |
| `POST /api/query` | Grounded text query (REST fallback) |
| `WS /ws/live` | WebSocket for real-time voice/video/text with Gemini Live API |

## Confidence Score Formula

Every response gets a real-time confidence score computed as:

```
C = 0.4 × Citation Coverage + 0.3 × Source Quality + 0.3 × Grounding Status

Where:
  Citation Coverage  = % of response backed by document sources
  Source Quality     = relevance score of retrieved chunks (0-100)
  Grounding Status   = 100 if grounded, 50 if partial, 0 if ungrounded
```

The confidence meter displays as an animated radial gauge with color coding:
- **Green (75-100%)** — High confidence, well-grounded
- **Amber (45-74%)** — Medium confidence, partial grounding
- **Red (0-44%)** — Low confidence, weak or no grounding

## Document Health Score

Each uploaded document receives a health grade:

```
H = 0.25 × Content Richness + 0.25 × Keyword Diversity
  + 0.25 × Embedding Coverage + 0.25 × Structure Quality

Grades: Excellent (80%+) | Good (60-79%) | Fair (40-59%) | Low (<40%)
```

## Premium Theme System

GroundTruth ships with two professionally designed themes:

- **Dark Mode** — Deep navy-charcoal backgrounds with ambient green/blue glow, optimized for low-light environments
- **Light Mode** — Color-graded warm ivory palette (`#f8f7f4` → `#f2f1ec` → `#eae9e4`), glass morphism header with `backdrop-filter: blur(12px)`, layered card depth with blue-tinted shadows, deeper green accent (`#16a34a`) for white-background contrast

Theme persists across sessions via `localStorage`. Toggle with the animated sun/moon switch in the header.

## Security

- Prompt injection detection (12 regex patterns)
- Document content treated as DATA, never as INSTRUCTIONS
- Input sanitization (control chars, length limits)
- Rate limiting per client
- Non-root Docker container
- No secrets in code (env-based config)

---

Built for the **Gemini Live Agent Challenge 2026** | [DevPost Submission](https://geminiliveagentchallenge.devpost.com/)
