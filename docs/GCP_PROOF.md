# GroundTruth — Google Cloud Platform Usage Proof

> This document demonstrates all Google Cloud services and APIs used by GroundTruth.

## 1. Gemini Live API (Real-Time Audio/Video)

**File:** [`backend/api/websocket_handler.py`](../backend/api/websocket_handler.py)

GroundTruth connects to Gemini Live API via `google-genai` SDK for real-time voice and vision interaction:

```python
# Line 55-56: Import Google GenAI types
from google.genai import types

# Line 165-191: Establish live streaming connection
def _connect_gemini(self):
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part(text=system_text)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self.settings.voice_name  # "Kore"
                )
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    return self.client.aio.live.connect(
        model=self.settings.gemini_live_model,  # gemini-2.5-flash-native-audio-preview
        config=config,
    )
```

**Live API operations used:**
- `client.aio.live.connect()` — Establish bidirectional streaming session
- `session.send_realtime_input(audio=...)` — Stream microphone audio (Line 324-329)
- `session.send_realtime_input(video=...)` — Stream camera frames (Line 335-340)
- `session.send_client_content(turns=...)` — Send grounded text prompts (Line 369-376)
- `session.receive()` — Receive audio responses + transcriptions (Line 197)
- Keep-alive heartbeat with silent PCM frames every 15s (Line 439-458)

---

## 2. Gemini Text Generation API (REST Fallback)

**File:** [`backend/api/query.py`](../backend/api/query.py)

Text-only queries use the Gemini REST API as a fallback when voice is not active:

```python
# Gemini text generation call
response = await gemini_client.aio.models.generate_content(
    model=settings.gemini_text_model,  # gemini-2.5-flash-preview-05-20
    contents=grounded_prompt,
    config=types.GenerateContentConfig(
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
    ),
)
```

---

## 3. Gemini Embedding API (Document Vectors)

**File:** [`backend/core/super_memory.py`](../backend/core/super_memory.py)

GroundTruth generates document embeddings using Gemini's embedding model for semantic search:

```python
# Line ~180: Generate embeddings via Gemini API
response = await self.gemini_client.aio.models.embed_content(
    model=self.embedding_model,  # gemini-embedding-001
    contents=text,
)
embedding = response.embeddings[0].values  # 3072-dimensional float vector
```

The embeddings are then **binary quantized** (32x compression):
```python
# 3072 floats → 384 bytes binary (sign-bit compression)
binary = bytes(
    sum(1 << (7 - j) for j in range(8) if vec[i*8 + j] > 0)
    for i in range(len(vec) // 8)
)
```

---

## 4. Google Cloud Run Deployment

**File:** [`deployment/deploy.sh`](../deployment/deploy.sh)

Production deployment to Google Cloud Run with these GCP APIs enabled:

```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com

gcloud run deploy groundtruth \
    --image="${IMAGE_NAME}:latest" \
    --platform=managed \
    --region="${REGION}" \
    --port=8080 \
    --memory=1Gi --cpu=2 \
    --set-env-vars="GOOGLE_API_KEY=${GOOGLE_API_KEY}"
```

---

## 5. Terraform Infrastructure-as-Code (GCP)

**File:** [`deployment/terraform/main.tf`](../deployment/terraform/main.tf)

Full IaC deployment with:
- `google_project_service` — Enables Cloud Run, Cloud Build, Artifact Registry, AI Platform APIs
- `google_artifact_registry_repository` — Docker image storage
- `google_cloud_run_v2_service` — Serverless container hosting
- `google_cloud_run_v2_service_iam_member` — Public access policy

---

## 6. Gemini Client Initialization

**File:** [`backend/app.py`](../backend/app.py)

```python
# Line 55-56: Initialize Google GenAI client
from google import genai
gemini_client = genai.Client(api_key=settings.google_api_key)
```

---

## 7. Configuration for Google Cloud

**File:** [`backend/config.py`](../backend/config.py)

```python
class Settings(BaseSettings):
    google_api_key: str          # Gemini API key
    google_cloud_project: str    # GCP project ID
    gcs_bucket_name: str         # Cloud Storage bucket
    google_genai_use_vertexai: bool = False  # Vertex AI toggle

    gemini_live_model: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    gemini_text_model: str = "gemini-2.5-flash-preview-05-20"
    embedding_model: str   = "gemini-embedding-001"
```

---

## 8. Docker Container (Cloud Run Ready)

**File:** [`Dockerfile`](../Dockerfile)

Multi-stage, non-root Docker image optimized for Cloud Run:
- Python 3.11 slim base
- Non-root `groundtruth` user (security best practice)
- Health check endpoint at `/api/health`
- Port 8080 (Cloud Run standard)

---

## Summary of Google Cloud APIs Used

| Google Cloud Service | SDK / API | Usage |
|---------------------|-----------|-------|
| **Gemini Live API** | `google-genai` → `client.aio.live.connect()` | Real-time voice + vision streaming |
| **Gemini Text API** | `google-genai` → `client.aio.models.generate_content()` | Text query fallback |
| **Gemini Embedding API** | `google-genai` → `client.aio.models.embed_content()` | Document vector embeddings |
| **Cloud Run** | `gcloud run deploy` / Terraform | Serverless container hosting |
| **Cloud Build** | `gcloud builds submit` | Container image building |
| **Artifact Registry** | Terraform resource | Docker image storage |
| **AI Platform API** | Enabled via `gcloud services enable` | Gemini model access |

---

> **All API calls use the `google-genai` Python SDK (PyPI: `google-genai`)** — Google's official Generative AI client library.
>
> Repository: https://github.com/BeeaxAI/Groundtruth
