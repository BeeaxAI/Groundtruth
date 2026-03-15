# GroundTruth — Demo Video Script (Under 4 Minutes)

## Pre-Recording Checklist
- [ ] GroundTruth running locally or on Cloud Run
- [ ] Prepare 2 sample documents (e.g., company policy PDF, product spec DOCX)
- [ ] Prepare 1 printed physical document for camera demo
- [ ] Clear browser, mic and camera permissions granted
- [ ] Screen recording tool ready (OBS recommended)
- [ ] Quiet room, good microphone

---

## Script

### [0:00 - 0:30] THE PROBLEM (Hook)

**Narration (voiceover with text slides):**

"Enterprise AI has a trust problem. When your AI assistant fabricates contract clauses, invents policy details, or cites documents that don't exist — the consequences aren't just embarrassing, they're costly and dangerous."

"Current AI chatbots can't guarantee their answers come from YOUR data. Until now."

**Visual:** Quick montage of AI hallucination examples → fade to GroundTruth logo.

---

### [0:30 - 1:00] THE SOLUTION

**Narration:**

"Meet GroundTruth — a zero-hallucination enterprise knowledge agent. Every answer is grounded in your source documents, every claim is cited, and every response is validated."

**Visual:** Show the GroundTruth UI — clean, dark interface with three panels.

"Let me show you how it works."

---

### [1:00 - 2:00] LIVE DEMO — VOICE + DOCUMENTS

**Action:** Upload a company policy PDF via drag-and-drop.

**Narration:**

"First, I upload our company's remote work policy. GroundTruth chunks it into sections and indexes it for retrieval."

**Visual:** Show document appearing in sidebar with chunk count.

**Action:** Click the microphone button and speak:

"What is our policy on remote work Fridays?"

**Visual:** Show:
1. Input transcription appearing ("You: What is our policy...")
2. Citation cards populating in the right panel
3. Agent response with [Source 1], [Source 2] citations inline
4. Green "✓ Grounded (2 sources cited)" badge appearing
5. Audio response playing through speakers

**Narration:**

"Notice: every claim has a citation tag pointing to the exact source. The grounding badge confirms the response was validated — no hallucination."

---

### [2:00 - 2:40] LIVE DEMO — CAMERA + HALLUCINATION RESISTANCE

**Action:** Turn on camera, hold a printed document in front of webcam.

**Narration:**

"GroundTruth can also see. I'm pointing my camera at a physical document — it reads the content and can answer questions about what it sees."

**Action:** Ask via voice: "What does this document say about the deadline?"

**Visual:** Show camera feed in right panel, response with citations.

**Action:** Now ask something NOT in any document:

"What was Apple's revenue last quarter?"

**Visual:** Show response: "The uploaded documents don't contain information about Apple's revenue. I can only answer based on the documents provided."

**Narration:**

"When I ask something outside the documents, GroundTruth refuses to guess. No hallucination. No made-up numbers. Just honesty."

---

### [2:40 - 3:20] ARCHITECTURE & SECURITY

**Visual:** Show the architecture diagram (Mermaid rendered).

**Narration:**

"Under the hood, GroundTruth uses a five-layer grounding architecture:

1. Strict system instructions prohibit general knowledge
2. Every query triggers document retrieval via BM25 scoring
3. Context is injected with mandatory citation requirements
4. Gemini generates responses through the Live API with real-time audio
5. A post-response validator checks every citation is real

The entire system runs on Google Cloud Run, built with the Google GenAI SDK, and includes prompt injection defenses — document content is always treated as data, never as instructions."

---

### [3:20 - 3:50] DEPLOYMENT & ENTERPRISE VALUE

**Visual:** Show terminal running `deploy.sh` or Terraform apply.

**Narration:**

"Deployment is fully automated with Terraform and gcloud scripts. One command deploys the complete stack to Cloud Run."

"For enterprises, GroundTruth means: no more AI liability from fabricated answers, full audit trails for compliance, and the confidence to deploy AI in regulated industries — healthcare, legal, finance."

---

### [3:50 - 4:00] CLOSING

**Visual:** GroundTruth logo + tagline.

**Narration:**

"GroundTruth. Because in enterprise AI, trust isn't optional — it's the product."

---

## Recording Tips

1. **Keep it moving** — no long pauses between scenes
2. **Show real interactions** — judges want to see LIVE software, not mockups
3. **Audio quality matters** — use a good mic, speak clearly
4. **Show the citation flow** — this is your differentiator, linger on it
5. **The hallucination resistance demo** — this is the money shot, make it clean
