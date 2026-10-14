# Task 2 — Model Integration Prototype

**Task:** Integrate a model, library, or public AI API into a small prototype.

This prototype upgrades the FAQ retrieval built in Task 1 (which used TF-IDF
keyword-overlap similarity) to real **semantic search**, powered by an
actual embedding model — not just word matching.

## Model used

- **Model:** [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) — a quantized ONNX port of the popular `sentence-transformers/all-MiniLM-L6-v2` sentence-embedding model, hosted publicly on the Hugging Face Hub.
- **Library:** [`@xenova/transformers`](https://www.npmjs.com/package/@xenova/transformers) (transformers.js) — runs the model locally via ONNX Runtime.
- **No API key required.** The model weights are downloaded from the public Hugging Face Hub on first run and cached locally (`.cache/`). There is no account, token, or paid service involved — this satisfies "no secret API keys" by construction, not just by omission.

## Why this over Task 1's TF-IDF approach

TF-IDF only matches shared *words*. A query like *"How much percentage do I
need in 12th grade to get in?"* shares almost no words with the FAQ
*"What is the minimum eligibility criteria for B.Tech admission?"* — TF-IDF
would likely miss it. A sentence embedding model captures *meaning*, so it
correctly matches the two as semantically equivalent (see example output
below, similarity 0.43).

## How it works

1. Load the `Xenova/all-MiniLM-L6-v2` feature-extraction pipeline.
2. Embed all 15 FAQ questions into 384-dimensional normalized vectors.
3. Embed the incoming user query the same way.
4. Compute cosine similarity between the query vector and every FAQ vector.
5. Return the best match if its similarity clears `0.35` (tuned from observed
   score distributions — see comment in `semantic_search.mjs`); otherwise
   return an honest "I don't have this information" fallback instead of
   guessing.

## Run it

```bash
cd task2-model-integration
npm install
npm run demo        # runs 4 built-in example queries
node semantic_search.mjs "Is there any scholarship for me?"   # ask your own question
```

## Example inputs and outputs

Run via `npm run demo`, output captured verbatim:

**Input:** `"How much percentage do I need in 12th grade to get in?"`
```json
{
  "query": "How much percentage do I need in 12th grade to get in?",
  "matched": true,
  "similarity": 0.4316,
  "matchedQuestion": "What is the minimum eligibility criteria for B.Tech admission?",
  "category": "eligibility",
  "answer": "Candidates must have passed 10+2 with Physics, Chemistry, and Mathematics with a minimum aggregate of 45% (40% for reserved categories)"
}
```

**Input:** `"Do I have to write any test to get admitted?"`
```json
{
  "query": "Do I have to write any test to get admitted?",
  "matched": true,
  "similarity": 0.7057,
  "matchedQuestion": "Is there an entrance exam required for admission?",
  "category": "eligibility",
  "answer": "Yes, admission is generally through UPSEE/AKTU counselling or a valid JEE Main score, followed by institute-level counselling"
}
```

**Input:** `"By when should I submit my application?"`
```json
{
  "query": "By when should I submit my application?",
  "matched": true,
  "similarity": 0.7185,
  "matchedQuestion": "What is the last date to submit the application form?",
  "category": "deadlines",
  "answer": "The last date for form submission is typically announced on the official admission portal; check the notice board for the current cycle's deadline"
}
```

**Input:** `"Can you tell me today's cricket score?"` (out-of-scope, guardrail test)
```json
{
  "query": "Can you tell me today's cricket score?",
  "matched": false,
  "similarity": 0.1753,
  "answer": "I don't have this information — please contact the admission cell."
}
```

Note the large gap between genuine-match scores (0.43–0.72) and the
out-of-scope score (0.18) — the model cleanly separates relevant from
irrelevant questions by meaning, not just keyword overlap.

## Files

- `semantic_search.mjs` — the prototype (model loading, embedding, similarity search, CLI)
- `faq_dataset.json` — same 15 FAQ pairs used in Task 1
- `package.json` — single dependency: `@xenova/transformers`
