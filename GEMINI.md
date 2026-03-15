# GroundTruth — Development Mandates

This document defines the foundational engineering standards and commit rules for the GroundTruth project. These rules are mandatory for all contributors and AI agents working on this codebase.

## 1. Commit Rules

All commits must adhere to the following standards:

### Format
*   **Imperative Mood:** Use the imperative mood in the subject line (e.g., "Add grounding validation" instead of "Added grounding validation").
*   **Subject Line:** Keep the first line concise (maximum 72 characters). Capitalize the first letter and do not end with a period.
*   **Body:** Use a blank line between the subject and the body. Use the body to explain **why** the change was made, especially for complex architectural shifts.
*   **Bullet Points:** Use hyphens for bulleted lists in the body to detail specific changes.

### Content
*   **Atomic Commits:** Each commit should represent a single logical change or feature. Avoid "mega-commits" that mix refactoring with new features.
*   **No WIPs:** Never commit broken code or "Work in Progress" states to the main branch.
*   **Verification:** Every commit must pass existing unit and E2E tests (`pytest`).
*   **Linting:** Commits must be free of major linting errors (F401, F841, E741).

### Examples
> **Good:**
> `Add BM25-based document retrieval to grounding engine`
>
> **Bad:**
> `fixed some bugs and added a file`

---

## 2. Engineering Standards

*   **Zero Hallucination:** Any change to the `grounding_engine.py` or retrieval logic must prioritize factual accuracy and source citation over conversational fluidity.
*   **Test-Driven Development:** New features must be accompanied by relevant unit tests in `backend/tests/` or E2E tests in `tests/e2e/`.
*   **Mock-First Testing:** Ensure that all tests can run in "mock mode" using `MockGeminiClient` to facilitate CI/CD without API dependencies.
*   **Type Safety:** Use Python type hints for all new function signatures and class definitions.
*   **Documentation:** Update `README.md` or relevant architecture docs when introducing new core modules or changing the system's public API.
