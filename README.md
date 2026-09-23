# AI Problem Design — Task 1 (SWYNEX Technologies Internship)

**Author:** Kushagra Gupta
**Task:** AI Problem Design
**Repository:** SWYNEX-AI-Problem-Design

---

## 1. Problem Statement

Prospective students and applicants (e.g., B.Tech admission seekers, internship applicants) repeatedly
ask the same handful of questions to college admission cells and internship coordinators — eligibility
criteria, deadlines, fee structure, document requirements, and application steps. These queries are
usually answered manually over email, WhatsApp, or phone, causing delays and repetitive effort for staff.

**Use case type:** Closed-domain Question Answering (Q&A) over a small, curated FAQ dataset.

## 2. Target User

- **Primary user:** A prospective student or internship applicant looking for quick, accurate answers
  about admissions/application processes.
- **Secondary user:** Admission cell staff / internship coordinators, who benefit from reduced repetitive
  query load.

## 3. Data Source

- A self-curated dataset of **40–60 question–answer pairs**, compiled from:
  - Publicly available admission/internship FAQ pages (e.g., AKTU/BBDITM admission notices, internship
    program guidelines).
  - Common questions observed in student WhatsApp/Telegram groups (rephrased, no personal data).
- Stored as a single structured file (`faq_dataset.csv` or `.jsonl`) with columns: `question`, `answer`,
  `category` (e.g., eligibility, fees, deadlines, documents).
- Dataset is small and static by design — no live scraping or PII, keeping the scope narrow and
  reproducible.

## 4. Approach

- Embed the FAQ dataset using a lightweight open-source embedding model.
- On a user query, retrieve the top-k most similar Q&A pairs (vector similarity search — a minimal RAG
  pipeline, no fine-tuning required).
- Pass the retrieved context + user query to an LLM to generate a natural-language answer grounded in the
  retrieved FAQ entries, rather than the model's own unguided knowledge.
- If no FAQ entry is sufficiently similar (below a similarity threshold), the system should respond with
  "I don't have this information — please contact the admission cell" rather than guessing.

## 5. Constraints

- **Data size:** Deliberately small (~50 entries) to keep the task scoped and demoable within the
  internship timeline.
- **Cost:** Must run on free-tier tools only (open-source embedding model, free-tier or locally-run LLM)
  — no paid API dependency.
- **Latency:** Response should return in under ~3 seconds for a good demo experience.
- **Hallucination risk:** The system must not fabricate policy details (fees, deadlines) — answers must
  be traceable to a specific FAQ entry, with an explicit fallback for out-of-scope questions.
- **No personal data:** Dataset must not include any real applicant's private information.

## 6. Evaluation Approach

- **Held-out test set:** Reserve ~10 questions (paraphrased versions of dataset questions, not seen
  verbatim) to check retrieval robustness to rewording.
- **Metrics:**
  - *Retrieval accuracy:* % of test questions for which the correct FAQ entry is retrieved in the top-3
    results.
  - *Answer relevance:* Manual 1–5 rating of whether the generated answer correctly reflects the
    retrieved FAQ content (no hallucinated details).
  - *Fallback correctness:* % of genuinely out-of-scope questions (e.g., "what's the weather today?")
    correctly met with the "I don't have this information" response instead of a fabricated answer.
- **Target bar for this task:** ≥80% retrieval accuracy and zero hallucinated factual claims (fees,
  dates, eligibility numbers) on the held-out set.

## 7. Deliverables

- This document (`README.md`).
- `faq_dataset.csv` — the curated FAQ dataset.
- Demo script or notebook showing a few example queries and responses.
- LinkedIn video walking through the problem framing and a live demo.

## 8. Live App

- **Live demo:** https://swynex-faq-assistant.netlify.app
- A deployable full-stack app lives in [`app/`](app/): a Netlify serverless function
  (`app/netlify/functions/ask.mts`) serves TF-IDF vector-similarity retrieval over the FAQ
  dataset with a similarity-threshold fallback, called by the static frontend in `app/public/`.
- A standalone client-side version (no backend) also lives in [`demo/`](demo/), used to record
  the LinkedIn video.

---

*Submitted as part of Task 1 of the SWYNEX Technologies internship — AI Problem Design.*
