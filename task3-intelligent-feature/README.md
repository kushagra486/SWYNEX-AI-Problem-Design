# Task 3 — Intelligent Feature, Evaluation & Final Project Package

**Task:** Improve the prototype with prompt or model evaluation examples,
failure cases, and a simple interface or notebook demo. Package the problem,
method, demo, limitations, and ethics notes into a final project README.

This folder is the final, packaged version of the SWYNEX admission-FAQ
assistant. It builds on Task 1 (TF-IDF retrieval) and Task 2 (semantic search
with `Xenova/all-MiniLM-L6-v2`) and adds the three things those didn't have:
an intelligent confidence-tiered response strategy, defensive error handling,
and a measured evaluation harness with documented failures.

---

## 1. Problem

Prospective students ask admission questions in their own words — not in the
exact phrasing of an FAQ page. A plain keyword search (Task 1's TF-IDF
baseline) misses paraphrases; a single-vector "best match" model (Task 2)
answers every query with equal confidence, even when it's actually guessing.
Neither tells the student when to trust the answer and when to ask a human
instead. Task 3 closes that gap.

## 2. Method

1. **Embedding model:** [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2), a quantized ONNX sentence-transformer, run locally via [`@xenova/transformers`](https://www.npmjs.com/package/@xenova/transformers) — no API key, no network calls after the first model download.
2. **Retrieval:** embed all 15 FAQ questions once; embed the incoming query; rank by cosine similarity.
3. **Intelligent feature — confidence tiers**, instead of a single matched/not-matched flag:
   - **High** (similarity ≥ 0.55) → answer directly.
   - **Medium** (0.35 ≤ similarity < 0.55) → answer, but say explicitly it's a best guess.
   - **Low** (similarity < 0.35) → don't guess. Say so, and surface the closest matching FAQ *categories* as a clarifying suggestion instead of a bare "I don't know."
4. **Error handling** around every place real input breaks naive code: empty/whitespace query, non-string input (e.g. a number), queries over 500 characters, a missing or corrupted FAQ dataset file, a failed model load, and an unexpected exception mid-REPL-session (caught per-turn so one bad input doesn't kill the session).
5. **Simple interface:** `assistant.mjs` runs as a demo (`--demo`), a one-shot CLI query, or an interactive REPL (`--repl`) you can actually converse with.
6. **Evaluation harness:** `eval.mjs` runs 15 labeled test cases (9 paraphrases, 3 genuinely out-of-scope queries, 3 deliberately adversarial edge cases) and reports pass/fail, similarity score, confidence tier, and overall accuracy.

Thresholds (0.35 / 0.55) were set from the Task 2 score-distribution analysis
and refined against this eval set — not hand-tuned per test case.

## 3. Demo

```bash
cd task3-intelligent-feature
npm install
npm run demo    # built-in demo queries, including deliberate error cases
npm run repl     # interactive chat interface
npm run eval     # labeled evaluation harness with pass/fail + accuracy
node assistant.mjs "Is there a scholarship for me?"   # single query
```

### Example output — confidence tiers in action

**High confidence** (direct answer):
```json
{
  "query": "What about fees for hostel seats?",
  "confidence": "high",
  "similarity": 0.6265,
  "runnerUpSimilarity": 0.2966,
  "matched": true,
  "matchedQuestion": "Is hostel accommodation available for outstation students?",
  "category": "documents",
  "answer": "Yes, hostel accommodation is available on a first-come-first-served basis; apply separately through the hostel admission form"
}
```
*(Note: this is actually a wrong match — see Limitations below. It's included here unedited because it's real output, not cherry-picked.)*

**Medium confidence** (hedged answer):
```json
{
  "query": "How much percentage do I need in 12th grade to get in?",
  "confidence": "medium",
  "similarity": 0.4316,
  "runnerUpSimilarity": 0.3827,
  "matched": true,
  "matchedQuestion": "What is the minimum eligibility criteria for B.Tech admission?",
  "category": "eligibility",
  "answer": "(best guess, please confirm) Candidates must have passed 10+2 with Physics, Chemistry, and Mathematics with a minimum aggregate of 45% (40% for reserved categories)"
}
```

**Low confidence** (clarifying fallback, not a bare refusal):
```json
{
  "query": "Can you tell me today's cricket score?",
  "confidence": "low",
  "similarity": 0.1753,
  "matched": false,
  "answer": "I don't have this information — please contact the admission cell.",
  "clarifyingSuggestion": "This might be related to: fees, deadlines. Try rephrasing, or ask about one of these."
}
```

**Error handling** (invalid input, no model call wasted):
```json
{ "query": "", "error": "Query is empty. Please ask an actual question." }
{ "query": 42, "error": "Query must be text, got number." }
```

### Evaluation results (real run, `npm run eval`)

15 labeled test cases — **13/15 correct (86.7%)**, confidence distribution: 8 high / 4 medium / 3 low.

| Query | Expected | Predicted | Confidence | Result |
|---|---|---|---|---|
| "How much percentage do I need in 12th grade to get in?" | eligibility | eligibility | medium (0.43) | PASS |
| "Do I have to write any test to get admitted?" | entrance exam | entrance exam | high (0.71) | PASS |
| "By when should I submit my application?" | deadline | deadline | high (0.72) | PASS |
| "What papers do I need to bring for admission?" | documents | documents | high (0.83) | PASS |
| "Is financial aid or scholarship available?" | scholarships | scholarships | high (0.81) | PASS |
| "How can I apply for an internship?" | internship application | internship application | high (0.90) | PASS |
| "Can I get my money back if I leave after joining?" | refund policy | refund policy | medium (0.49) | PASS |
| "How long does the internship last?" | internship duration | internship duration | high (0.88) | PASS |
| "Is there a hostel for students from other cities?" | hostel | hostel | high (0.76) | PASS |
| "What's today's weather like?" | (no match) | (no match) | low (0.02) | PASS |
| "Can you tell me today's cricket score?" | (no match) | (no match) | low (0.18) | PASS |
| "fee" (single word) | course fee | course fee | medium (0.46) | PASS |
| "...12th marksheet, now what documents are left?" | documents | documents | high (0.55) | PASS |
| **"Write me a poem about admissions."** | (no match) | documents | medium (0.44) | **FAIL** |
| **"money"** (underspecified) | course fee | (no match) | low (0.13) | **FAIL** |

## 4. Failure cases (honest, not cherry-picked)

1. **False positive — off-topic phrasing that borrows domain words.**
   *"Write me a poem about admissions."* is not a real question, but it
   shares enough vocabulary with the FAQ corpus ("admissions") that the
   embedding model scores it at medium confidence (0.44) and the assistant
   answers as if it were a genuine query. **Why:** sentence embeddings
   capture topical similarity, not intent — they can't tell "ask about X"
   from "mention X." A production system would need an intent classifier
   or instruction-filtering step in front of retrieval.
2. **False negative — single ambiguous word.**
   *"money"* alone scores only 0.13 (low confidence → fallback), even
   though a human would reasonably read it as a fee question. **Why:** a
   1-word query carries very little semantic signal for a sentence-level
   embedding model trained on full sentences; it doesn't fail gracefully
   on extremely short input the way longer paraphrases do.
3. **Plausible-looking wrong match (see the "high confidence" example above).**
   *"What about fees for hostel seats?"* matched the hostel-accommodation
   FAQ instead of a fee-related one, with high confidence (0.63) — the
   model latched onto "hostel" and ignored "fees." This is the most
   concerning failure mode: it's wrong *and* confident, which is exactly
   what the confidence-tier system cannot catch, since the model's
   similarity score and its actual correctness aren't the same thing.

These are reported as-is, including the one genuinely embarrassing case,
because an evaluation that only shows passing examples isn't an evaluation.

## 5. Limitations

- **Confidence ≠ correctness.** The tier system reduces *overconfident
  wrong answers on clearly off-topic input*, but as failure case 3 shows,
  it cannot catch a wrong answer the model is confident about. Confidence
  tiers are a mitigation, not a guarantee.
- **Single-sentence embeddings struggle with very short or multi-intent
  queries** (failure cases 1 and 2).
- **Fixed FAQ corpus.** The assistant can only answer questions whose
  intent is already represented in `faq_dataset.json` (15 entries); it has
  no way to answer genuinely novel questions correctly, by design — it
  falls back instead of fabricating an answer (see Ethics).
- **No conversation memory.** Each query is evaluated independently; the
  REPL doesn't use prior turns as context.
- **Evaluation set is small (15 cases).** It's large enough to surface real
  failure modes but too small to produce a statistically precise accuracy
  figure; treat 86.7% as indicative, not a tight bound.

## 6. Ethics notes

- **No fabrication by design.** The low-confidence fallback exists
  specifically so the assistant says "I don't know, here's a human to
  contact" rather than generating a plausible-sounding but wrong answer to
  a question outside its FAQ corpus — important for admissions, where a
  wrong answer about eligibility or deadlines has real consequences for a
  student's application.
- **No personal data collected or stored.** Queries are processed in
  memory for a single request/REPL turn and never logged, persisted, or
  sent to a third party — the model runs entirely locally.
- **No API key / no paid service** — nothing here depends on a vendor
  account, usage billing, or external inference API that could introduce a
  privacy or cost dependency for downstream users of this prototype.
- **Known bias risk:** the FAQ corpus itself (15 hand-written entries) is a
  small, English-only sample and does not represent every phrasing a real
  applicant might use (dialectal English, non-native phrasing, etc.); the
  evaluation set is similarly English-only and was not built to measure
  fairness across applicant demographics. A production deployment would
  need a much larger, more representative query set before being trusted
  for real admissions decisions.
- **Appropriate use:** this is an informational FAQ aid, not a decision-maker.
  It should never be the sole channel for eligibility, deadline, or
  fee-waiver determinations — the fallback answer always points back to the
  human admission cell for exactly this reason.

## 7. Files

- `assistant.mjs` — the intelligent assistant: confidence tiers, error
  handling, demo/CLI/REPL interface.
- `eval.mjs` — labeled evaluation harness (15 cases, including adversarial ones).
- `faq_dataset.json` — same 15 FAQ pairs used in Tasks 1 and 2.
- `package.json` — single dependency: `@xenova/transformers`.

## 8. Relationship to Tasks 1 and 2

| | Task 1 | Task 2 | Task 3 (this folder) |
|---|---|---|---|
| Retrieval | TF-IDF keyword overlap | Semantic embeddings | Semantic embeddings |
| Response | matched / not-matched | matched / not-matched | high / medium / low confidence |
| Error handling | — | — | validated input, dataset, model load |
| Evaluation | — | 4 demo queries | 15 labeled cases, measured accuracy, documented failures |
| Interface | web app | CLI | CLI + interactive REPL |
