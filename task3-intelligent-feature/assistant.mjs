// Task 3 — Intelligent feature + error handling + simple interface
//
// Builds on the Task 2 semantic-search prototype (Xenova/all-MiniLM-L6-v2 via
// transformers.js) and adds the pieces Task 2 deliberately left out:
//
//   1. An intelligent confidence-tiered response strategy, instead of a
//      single matched/not-matched boolean. A model's similarity score is a
//      continuum, not a certainty — a good assistant should say so.
//   2. Defensive error handling around every place real input can break
//      things: empty input, non-string input, a missing/corrupt dataset
//      file, and a failed model load.
//   3. A simple interactive interface (a REPL), not just one-shot CLI calls.
//
// See eval.mjs for the labeled evaluation harness and documented failure
// cases, and README.md for the write-up (problem, method, limitations, ethics).

import { pipeline, cos_sim } from "@xenova/transformers";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import readline from "node:readline";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Confidence tiers, tuned from the same score distribution analysis as
// Task 2 (see eval.mjs for the data backing these cutoffs):
//   HIGH   (>= 0.55): answer directly, no hedging.
//   MEDIUM (0.35-0.55): answer, but say explicitly that it's a best guess.
//   LOW    (< 0.35): don't guess — say so, and surface the closest FAQ
//                     categories as a "did you mean" clarification instead
//                     of silently failing.
const HIGH_CONFIDENCE = 0.55;
const LOW_CONFIDENCE = 0.35;

class FaqAssistantError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = "FaqAssistantError";
    this.cause = cause;
  }
}

async function loadFaqDataset() {
  const filePath = path.join(__dirname, "faq_dataset.json");
  let raw;
  try {
    raw = await readFile(filePath, "utf-8");
  } catch (err) {
    throw new FaqAssistantError(
      `Could not read FAQ dataset at ${filePath}. Is the file missing?`,
      err
    );
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new FaqAssistantError(
      `FAQ dataset at ${filePath} is not valid JSON — it may be corrupted.`,
      err
    );
  }
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new FaqAssistantError(
      "FAQ dataset must be a non-empty array of {question, answer, category} objects."
    );
  }
  for (const [i, entry] of parsed.entries()) {
    if (!entry || typeof entry.question !== "string" || typeof entry.answer !== "string") {
      throw new FaqAssistantError(
        `FAQ dataset entry at index ${i} is malformed (missing question/answer).`
      );
    }
  }
  return parsed;
}

async function loadModel() {
  try {
    return await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  } catch (err) {
    throw new FaqAssistantError(
      "Failed to load the embedding model (Xenova/all-MiniLM-L6-v2). " +
        "This is usually a network problem on first run, since the model " +
        "weights are fetched from the Hugging Face Hub and cached locally.",
      err
    );
  }
}

async function embed(extractor, text) {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

async function buildIndex(extractor, faqs) {
  const vectors = [];
  for (const faq of faqs) {
    vectors.push(await embed(extractor, faq.question));
  }
  return vectors;
}

// The intelligent feature: rank every FAQ by similarity, then decide HOW to
// respond (direct answer / hedged answer / clarifying question) based on
// where the best score (and the runner-up) land, instead of a single
// matched/not-matched cutoff.
function classifyConfidence(bestScore) {
  if (bestScore >= HIGH_CONFIDENCE) return "high";
  if (bestScore >= LOW_CONFIDENCE) return "medium";
  return "low";
}

async function answerQuery(extractor, faqs, vectors, query) {
  // --- error handling: validate input before touching the model ---
  if (query === undefined || query === null) {
    return { query, error: "Query is missing. Please provide a question." };
  }
  if (typeof query !== "string") {
    return { query, error: `Query must be text, got ${typeof query}.` };
  }
  const trimmed = query.trim();
  if (trimmed.length === 0) {
    return { query, error: "Query is empty. Please ask an actual question." };
  }
  if (trimmed.length > 500) {
    return {
      query,
      error: "Query is too long (max 500 characters) — please ask one question at a time.",
    };
  }

  let queryVector;
  try {
    queryVector = await embed(extractor, trimmed);
  } catch (err) {
    return { query, error: `Embedding failed: ${err.message}` };
  }

  const scored = vectors
    .map((v, i) => ({ i, score: cos_sim(queryVector, v) }))
    .sort((a, b) => b.score - a.score);

  const best = scored[0];
  const runnerUp = scored[1];

  if (!best || !Number.isFinite(best.score)) {
    return { query, error: "Similarity computation failed (no valid score produced)." };
  }

  const confidence = classifyConfidence(best.score);
  const bestFaq = faqs[best.i];

  if (confidence === "low") {
    // Intelligent fallback: don't just say "I don't know" — offer the two
    // closest categories as a clarifying prompt, since the best score may
    // still reflect a loosely related topic even though it's below threshold.
    const suggestions = scored
      .slice(0, 2)
      .map((s) => faqs[s.i].category)
      .filter((c, idx, arr) => arr.indexOf(c) === idx);
    return {
      query,
      confidence,
      similarity: Number(best.score.toFixed(4)),
      matched: false,
      answer: "I don't have this information — please contact the admission cell.",
      clarifyingSuggestion:
        suggestions.length > 0
          ? `This might be related to: ${suggestions.join(", ")}. Try rephrasing, or ask about one of these.`
          : undefined,
    };
  }

  return {
    query,
    confidence,
    similarity: Number(best.score.toFixed(4)),
    runnerUpSimilarity: runnerUp ? Number(runnerUp.score.toFixed(4)) : undefined,
    matched: true,
    matchedQuestion: bestFaq.question,
    category: bestFaq.category,
    answer:
      confidence === "medium"
        ? `(best guess, please confirm) ${bestFaq.answer}`
        : bestFaq.answer,
  };
}

const DEMO_QUERIES = [
  "How much percentage do I need in 12th grade to get in?", // high confidence
  "What about fees for hostel seats?",                        // medium confidence, ambiguous between fees/documents
  "Can you tell me today's cricket score?",                   // low confidence -> clarifying fallback
  "",                                                          // error case: empty input
  42,                                                          // error case: non-string input
];

async function runDemo(extractor, faqs, vectors) {
  for (const query of DEMO_QUERIES) {
    const result = await answerQuery(extractor, faqs, vectors, query);
    console.log(JSON.stringify(result, null, 2));
    console.log("---");
  }
}

async function runRepl(extractor, faqs, vectors) {
  console.log("Intelligent FAQ assistant — type a question, or 'exit' to quit.\n");
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ask = () =>
    new Promise((resolve) => rl.question("> ", resolve));

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const line = await ask();
    if (line.trim().toLowerCase() === "exit") break;
    try {
      const result = await answerQuery(extractor, faqs, vectors, line);
      console.log(JSON.stringify(result, null, 2));
    } catch (err) {
      // Defensive: even an unexpected exception during a turn shouldn't
      // crash the REPL — report it and keep the session alive.
      console.error(`Unexpected error, continuing session: ${err.message}`);
    }
  }
  rl.close();
}

async function main() {
  let faqs;
  try {
    faqs = await loadFaqDataset();
  } catch (err) {
    console.error(`Startup failed: ${err.message}`);
    process.exit(1);
  }

  console.log("Loading model: Xenova/all-MiniLM-L6-v2 (public, no API key)...\n");
  let extractor;
  try {
    extractor = await loadModel();
  } catch (err) {
    console.error(`Startup failed: ${err.message}`);
    process.exit(1);
  }

  console.log(`Embedding ${faqs.length} FAQ questions...\n`);
  const vectors = await buildIndex(extractor, faqs);

  const isDemo = process.argv.includes("--demo");
  const isRepl = process.argv.includes("--repl");
  const cliQuery = process.argv.slice(2).filter((a) => !a.startsWith("--")).join(" ");

  if (isRepl) {
    await runRepl(extractor, faqs, vectors);
  } else if (isDemo || !cliQuery) {
    await runDemo(extractor, faqs, vectors);
  } else {
    const result = await answerQuery(extractor, faqs, vectors, cliQuery);
    console.log(JSON.stringify(result, null, 2));
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
