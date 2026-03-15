# GroundTruth — Submission Checklist

## Required Submissions

- [ ] **Text Description** — Project features, technologies, data sources, findings/learnings
  - Covered in README.md — adapt for DevPost form

- [ ] **Public Code Repository** — With reproducible spin-up instructions
  - Push to GitHub: `https://github.com/YOUR_USERNAME/groundtruth`
  - Verify README has clear setup instructions
  - Verify `.env.example` exists (no real keys committed!)

- [ ] **Google Cloud Deployment Proof** — Screen recording or code showing GCP usage
  - Option A: Screen record `deploy.sh` running successfully
  - Option B: Include `deployment/deploy.sh` and `deployment/terraform/main.tf`
  - Show Cloud Run console with service running

- [ ] **Architecture Diagram** — Visual system representation
  - Render `docs/architecture.mermaid` to PNG/SVG
  - Use https://mermaid.live/ to render and screenshot

- [ ] **Demo Video** — Under 4 minutes, live features, no mockups
  - Follow `docs/DEMO_SCRIPT.md`
  - Upload to YouTube (unlisted) or Loom
  - Must show: voice interaction, camera, citations, hallucination resistance

## Bonus Points (Easy Wins!)

- [ ] **Blog/Content** — Publish about building with Google AI/Cloud
  - Write a Medium or Dev.to post
  - Title suggestion: "Building a Zero-Hallucination AI Agent with Gemini Live API"
  - Use hashtag: #GeminiLiveAgentChallenge
  - Include architecture diagram and code snippets

- [ ] **Automated Cloud Deployment** — Scripts or IaC
  - Already done! `deployment/deploy.sh` + `deployment/terraform/`

- [ ] **Google Developer Group** — Join with public profile
  - Go to: https://developers.google.com/community/gdg
  - Join a local GDG
  - Include profile link in submission

## Pre-Submission Verification

- [ ] Code runs locally with `python main.py`
- [ ] All 3 modalities work: voice, camera, text
- [ ] Document upload works for PDF, DOCX, TXT
- [ ] Citations appear in responses
- [ ] Grounding badge shows "✓ Grounded"
- [ ] Hallucination resistance works (refuses to answer outside documents)
- [ ] No API keys in committed code
- [ ] README is clear and complete
- [ ] Demo video is under 4 minutes
- [ ] Video shows LIVE software (not mockups)

## DevPost Submission Fields

| Field | Content |
|-------|---------|
| Project Name | GroundTruth |
| Tagline | Zero-Hallucination Enterprise Knowledge Agent |
| Category | Live Agents |
| Built With | Gemini Live API, Google GenAI SDK, Python, FastAPI, Google Cloud Run, Terraform |
| Try It Out | [Cloud Run URL] |
| GitHub | [Repository URL] |
| Video | [YouTube/Loom URL] |

## Timeline

| Task | Deadline |
|------|----------|
| Code complete + tested | March 15, evening |
| Demo video recorded | March 16, morning |
| Blog post published | March 16, morning |
| DevPost submission | March 16, before 5:00 PM PDT |
