"""
Phases 11–14: WebSocket handler for real-time voice + vision interaction.

Phase 11: Connection lifecycle management
Phase 12: Gemini Live API session handler
Phase 13: Audio streaming pipeline (16kHz in → 24kHz out)
Phase 14: Video frame pipeline (JPEG @ ≤1 FPS)

Protocol (client → server):
  {"type": "audio", "data": "<base64 PCM 16kHz>"}
  {"type": "video", "data": "<base64 JPEG>"}
  {"type": "text",  "text": "user question"}
  {"type": "context_inject", "query": "..."}
  {"type": "ping"}

Protocol (server → client):
  {"type": "audio", "data": "<base64 PCM 24kHz>"}
  {"type": "transcript_input", "text": "..."}
  {"type": "transcript_output", "text": "...", "full_text": "..."}
  {"type": "citations", "citations": [...], "has_context": bool}
  {"type": "turn_complete", "validation": {...}}
  {"type": "interrupted"}
  {"type": "status", "message": "..."}
  {"type": "error", "message": "..."}
  {"type": "pong"}
"""

import asyncio
import base64
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

logger = logging.getLogger(__name__)

MAX_RECONNECTS = 5


class LiveSessionHandler:
    """
    Manages a single WebSocket client session with Gemini Live API.
    Handles audio streaming, video frames, text queries, and grounding.
    Auto-reconnects to Gemini when the session expires.
    Pre-loads document context so voice queries can reference documents.
    """

    def __init__(
        self, ws: WebSocket, gemini_client, doc_service,
        grounding_engine, settings
    ):
        self.ws = ws
        self.client = gemini_client
        self.doc_service = doc_service
        self.grounding = grounding_engine
        self.settings = settings

        self.session = None
        self._receive_task: Optional[asyncio.Task] = None
        self._accumulated_transcript = ""
        self._last_citations = []
        self._last_user_query = ""
        self._active = False
        self._session_alive = False
        self._state_lock = asyncio.Lock()  # protects transcript + citations

        self._MAX_TRANSCRIPT_CHARS = 50_000  # 50KB cap to prevent OOM

    async def run(self):
        """Main session lifecycle."""
        await self.ws.accept()
        logger.info("WebSocket client connected")

        if not self.client:
            await self._send({
                "type": "error",
                "message": "Gemini not configured. Set GOOGLE_API_KEY."
            })
            await self.ws.close()
            return

        try:
            await self._session_loop()
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket session error: {e}", exc_info=True)
            await self._send_safe({"type": "error", "message": "Session error. Please refresh."})
        finally:
            await self._cleanup()

    # ---- Session Loop with Auto-Reconnect ----

    async def _session_loop(self):
        """Connect to Gemini with auto-reconnect on session expiry."""
        reconnect_count = 0

        while reconnect_count <= MAX_RECONNECTS:
            try:
                async with self._connect_gemini() as session:
                    self.session = session
                    self._active = True
                    self._session_alive = True
                    reconnect_count = 0
                    await self._send({"type": "status", "message": "connected"})

                    self._receive_task = asyncio.create_task(
                        self._receive_from_gemini()
                    )
                    client_task = asyncio.create_task(
                        self._receive_from_client()
                    )
                    keepalive_task = asyncio.create_task(
                        self._keepalive_loop()
                    )

                    done, pending = await asyncio.wait(
                        [self._receive_task, client_task, keepalive_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass

                    if client_task in done:
                        exc = client_task.exception()
                        if exc:
                            raise exc
                        return

                    reconnect_count += 1
                    logger.info(
                        f"Gemini session expired, auto-reconnecting "
                        f"({reconnect_count}/{MAX_RECONNECTS})..."
                    )
                    # Don't clear _last_citations here — they persist for
                    # the current query's turn_complete validation.
                    # Only clear transcript (Gemini restarts fresh).
                    self._accumulated_transcript = ""
                    await self._send_safe({"type": "status", "message": "reconnecting"})
                    await asyncio.sleep(0.5)

            except WebSocketDisconnect:
                raise
            except Exception as e:
                reconnect_count += 1
                logger.error(
                    f"Session error, reconnecting "
                    f"({reconnect_count}/{MAX_RECONNECTS}): {e}"
                )
                if reconnect_count > MAX_RECONNECTS:
                    await self._send_safe({
                        "type": "error",
                        "message": "Connection lost. Please refresh."
                    })
                    break
                await self._send_safe({"type": "status", "message": "reconnecting"})
                await asyncio.sleep(1)

    # ---- Phase 12: Gemini Live API Connection ----

    def _connect_gemini(self):
        """Return the Gemini Live API async context manager."""
        system_text = self.grounding.build_system_instruction(
            self.doc_service.get_document_summary()
        )

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(
                parts=[types.Part(text=system_text)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.settings.voice_name
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        logger.info("Connecting to Gemini Live API...")
        return self.client.aio.live.connect(
            model=self.settings.gemini_live_model,
            config=config,
        )

    # ---- Phase 13 & 14: Receive from Gemini (audio + transcriptions) ----

    async def _receive_from_gemini(self):
        try:
            async for response in self.session.receive():
                if not self._active:
                    break

                content = response.server_content
                if not content:
                    continue

                # Audio response chunks
                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data:
                            audio_b64 = base64.b64encode(
                                part.inline_data.data).decode()
                            await self._send({"type": "audio", "data": audio_b64})

                # Output transcription (what Gemini says)
                if content.output_transcription and content.output_transcription.text:
                    async with self._state_lock:
                        self._accumulated_transcript += content.output_transcription.text
                        # Cap to prevent unbounded memory growth on long sessions
                        if len(self._accumulated_transcript) > self._MAX_TRANSCRIPT_CHARS:
                            self._accumulated_transcript = self._accumulated_transcript[-self._MAX_TRANSCRIPT_CHARS:]
                        full = self._accumulated_transcript
                    await self._send({
                        "type": "transcript_output",
                        "text": content.output_transcription.text,
                        "full_text": full,
                    })

                # Input transcription (what user said)
                if content.input_transcription and content.input_transcription.text:
                    user_text = content.input_transcription.text
                    self._last_user_query = user_text
                    await self._send({"type": "transcript_input", "text": user_text})
                    # Note: Do NOT call _inject_context here — sending client_content
                    # while audio is streaming causes session interruption/crash.
                    # Document context is pre-loaded at session start instead.

                # Turn complete
                if content.turn_complete:
                    await self._handle_turn_complete()

                # Interrupted by user
                if content.interrupted:
                    async with self._state_lock:
                        self._accumulated_transcript = ""
                        self._last_citations = []
                    await self._send({"type": "interrupted"})

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Gemini receive error: {e}", exc_info=True)
            self._session_alive = False

    async def _handle_turn_complete(self):
        has_docs = self.doc_service.has_documents()

        async with self._state_lock:
            transcript = self._accumulated_transcript
            citations = list(self._last_citations)
            user_query = self._last_user_query

        if transcript and citations:
            # Record citation hits for hallucination heatmap
            for c in citations:
                self.doc_service.record_citation(c.chunk_id)

            # We have both a response and citations — validate grounding
            validation = self.grounding.validate_response(transcript, citations)
            confidence = self.grounding.compute_confidence(validation, citations)
            self.grounding.log_query(transcript, citations, True, validation, transcript)

            validation_dict = validation.to_dict()
            validation_dict["confidence"] = confidence

            # Record knowledge gap if confidence is low or status is ungrounded
            gap_query = user_query or transcript
            conf_score = confidence.get("score", 100) if isinstance(confidence, dict) else confidence
            gap_status = validation_dict.get("status", "")
            self.doc_service.record_gap(
                query=gap_query,
                confidence_score=conf_score,
                status=gap_status,
            )

            # Generate smart follow-up suggestions
            follow_ups = self.doc_service.generate_follow_ups(transcript, citations)

            await self._send({
                "type": "turn_complete",
                "validation": validation_dict,
                "follow_ups": follow_ups,
            })
        elif transcript and has_docs:
            # Gemini responded, docs exist but no specific citations matched.
            gap_query = user_query or transcript
            self.doc_service.record_gap(
                query=gap_query, confidence_score=10, status="no_match",
            )
            await self._send({"type": "turn_complete", "validation": {
                "status": "no_match", "valid": True,
                "reason": "Documents uploaded but no specific match for this query",
                "confidence": {"score": 10, "level": "low", "breakdown": {}},
            }})
        else:
            # No documents uploaded at all
            status = "no_match" if has_docs else "no_context"
            reason = ("No matching content found in uploaded documents"
                      if has_docs else "No documents uploaded")
            gap_query = user_query or transcript
            if gap_query:
                self.doc_service.record_gap(
                    query=gap_query, confidence_score=0, status=status,
                )
            await self._send({"type": "turn_complete", "validation": {
                "status": status, "valid": True, "reason": reason,
                "confidence": {"score": 0, "level": "none", "breakdown": {}},
            }})

        async with self._state_lock:
            self._accumulated_transcript = ""
            self._last_citations = []
            self._last_user_query = ""

    # ---- Phase 13: Audio Streaming (Client → Gemini) ----

    async def _handle_audio(self, data_b64: str):
        audio_bytes = base64.b64decode(data_b64)
        await self.session.send_realtime_input(
            audio=types.Blob(
                data=audio_bytes,
                mime_type=f"audio/pcm;rate={self.settings.input_sample_rate}",
            )
        )

    # ---- Phase 14: Video Frame Streaming (Client → Gemini) ----

    async def _handle_video(self, data_b64: str):
        frame_bytes = base64.b64decode(data_b64)
        await self.session.send_realtime_input(
            video=types.Blob(
                data=frame_bytes,
                mime_type="image/jpeg",
            )
        )

    # ---- Text Query with Context Injection ----

    async def _handle_text(self, text: str):
        from utils.security import InputSanitizer
        sanitizer = InputSanitizer()
        clean_text, _ = sanitizer.sanitize_query(text)
        if not clean_text:
            return

        if not self._session_alive or (self._receive_task and self._receive_task.done()):
            self._session_alive = False
            await self._send({"type": "error", "message": "Reconnecting to Gemini..."})
            await self._send({"type": "turn_complete", "validation": {
                "status": "no_context", "valid": False, "reason": "Session reconnecting"
            }})
            return

        async with self._state_lock:
            self._last_user_query = clean_text
        relevant = await self.doc_service.search_hybrid(clean_text, top_k=self.settings.max_retrieval_chunks)
        grounded_prompt, citations, has_context = self.grounding.build_grounded_prompt(
            clean_text, relevant, max_context_chars=self.settings.max_context_chars,
            has_documents=self.doc_service.has_documents(),
        )

        async with self._state_lock:
            self._last_citations = citations

        try:
            await asyncio.wait_for(
                self.session.send_client_content(
                    turns=[{"role": "user", "parts": [
                        {"text": grounded_prompt}]}],
                    turn_complete=True,
                ),
                timeout=15.0,
            )
            logger.info(f"Sent text query to Gemini: '{clean_text[:60]}'")
        except asyncio.TimeoutError:
            logger.error(
                "Timeout sending text to Gemini — session likely dead")
            self._session_alive = False
            await self._send({"type": "error", "message": "Session timed out. Reconnecting..."})
            await self._send({"type": "turn_complete", "validation": {
                "status": "no_context", "valid": False, "reason": "Session timed out"
            }})
            return
        except Exception as e:
            logger.error(f"Failed to send text to Gemini: {e}")
            self._session_alive = False
            await self._send({"type": "error", "message": "Reconnecting to Gemini..."})
            await self._send({"type": "turn_complete", "validation": {
                "status": "no_context", "valid": False, "reason": str(e)
            }})
            return

        if citations:
            await self._send({
                "type": "citations",
                "citations": [c.to_dict() for c in citations],
                "has_context": has_context,
            })

    async def _inject_context(self, user_speech: str):
        """Inject additional document context when user speaks."""
        if not self.doc_service.has_documents():
            return

        from utils.security import InputSanitizer
        clean_speech, _ = InputSanitizer().sanitize_query(user_speech)
        if not clean_speech:
            return

        relevant = self.doc_service.search(
            clean_speech, top_k=self.settings.max_retrieval_chunks)
        if not relevant:
            return

        grounded_prompt, citations, has_context = self.grounding.build_grounded_prompt(
            user_speech, relevant, max_context_chars=self.settings.max_context_chars,
            has_documents=True,
        )

        async with self._state_lock:
            self._last_citations = citations

        if has_context:
            try:
                await self.session.send_client_content(
                    turns=[{
                        "role": "user",
                        "parts": [{"text": f"[DOCUMENT CONTEXT — use these sources to answer]\n{grounded_prompt}"}],
                    }],
                    turn_complete=False,
                )
                await self._send({
                    "type": "citations",
                    "citations": [c.to_dict() for c in citations],
                    "has_context": True,
                })
            except Exception as e:
                logger.warning(f"Failed to inject context: {e}")

    # ---- Keep-alive to prevent Gemini session timeout ----

    async def _keepalive_loop(self):
        """Send periodic silent pings to Gemini to prevent session expiry."""
        try:
            while self._active and self._session_alive:
                await asyncio.sleep(15)  # ping every 15 seconds
                if self.session and self._session_alive:
                    try:
                        # Send an empty audio frame as heartbeat
                        # (silent PCM = all zeros, minimal size)
                        silent_frame = b'\x00' * 320  # 10ms of 16kHz PCM silence
                        await self.session.send_realtime_input(
                            audio=types.Blob(
                                data=silent_frame,
                                mime_type=f"audio/pcm;rate={self.settings.input_sample_rate}",
                            )
                        )
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

    # ---- Main Client Message Router ----

    # 10 MB default; overridden by settings.max_ws_message_bytes if available
    _MAX_MSG_BYTES = 10 * 1024 * 1024

    async def _receive_from_client(self):
        max_bytes = getattr(self.settings, "max_ws_message_bytes", self._MAX_MSG_BYTES)
        while self._active:
            raw = await self.ws.receive_text()

            if len(raw.encode()) > max_bytes:
                await self._send({"type": "error", "message": "Message too large."})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send({"type": "error", "message": "Invalid message format."})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio":
                await self._handle_audio(msg["data"])
            elif msg_type == "video":
                await self._handle_video(msg["data"])
            elif msg_type == "text":
                await self._handle_text(msg.get("text", ""))
            elif msg_type == "context_inject":
                await self._inject_context(msg.get("query", ""))
            elif msg_type == "doc_update":
                # Trigger session reconnect so the system instruction
                # includes the new document content
                logger.info("Document updated — triggering session refresh")
                self._session_alive = False
                if self._receive_task:
                    self._receive_task.cancel()
            elif msg_type == "ping":
                await self._send({"type": "pong"})

    # ---- Utility ----

    async def _send(self, data: dict):
        await self.ws.send_json(data)

    async def _send_safe(self, data: dict):
        try:
            await self.ws.send_json(data)
        except Exception:
            pass

    async def _cleanup(self):
        self._active = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        self.session = None
        logger.info("WebSocket session cleaned up")
