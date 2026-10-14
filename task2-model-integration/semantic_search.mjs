// Task 2 — Model integration prototype
//
// Integrates a public Hugging Face model (Xenova/all-MiniLM-L6-v2, a quantized
// ONNX port of sentence-transformers/all-MiniLM-L6-v2) via the transformers.js
// library. The model is downloaded from the public Hugging Face Hub on first
// run and cached locally — no API key, no account, no paid service.
//
// This replaces the TF-IDF vector-similarity retrieval used in Task 1 with
// real sentence embeddings, so paraphrases and synonyms that share almost no
// words with the original FAQ question can still be matched correctly.

import { pipeline, cos_sim } from "@xenova/transformers";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Tuned from observed scores: correct paraphrase matches scored 0.43-0.81,
// while genuinely out-of-scope queries scored 0.03-0.18 (see debug_scores.mjs).
// 0.35 sits cleanly in the gap between those two clusters.
const SIMILARITY_THRESHOLD = 0.35;

async function loadFaqDataset() {
  const raw = await readFile(path.join(__dirname, "faq_dataset.json"), "utf-8");
  return JSON.parse(raw);
}

async function buildIndex(extractor, faqs) {
  const vectors = [];
  for (const faq of faqs) {
    const output = await extractor(faq.question, { pooling: "mean", normalize: true });
    vectors.push(Array.from(output.data));
  }
  return vectors;
}

async function answerQuery(extractor, faqs, vectors, query) {
  const queryOutput = await extractor(query, { pooling: "mean", normalize: true });
  const queryVector = Array.from(queryOutput.data);

  let bestIndex = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < vectors.length; i++) {
    const score = cos_sim(queryVector, vectors[i]);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  if (bestScore < SIMILARITY_THRESHOLD) {
    return {
      query,
      matched: false,
      similarity: Number(bestScore.toFixed(4)),
      answer: "I don't have this information — please contact the admission cell.",
    };
  }

  const match = faqs[bestIndex];
  return {
    query,
    matched: true,
    similarity: Number(bestScore.toFixed(4)),
    matchedQuestion: match.question,
    category: match.category,
    answer: match.answer,
  };
}

const DEMO_QUERIES = [
  "How much percentage do I need in 12th grade to get in?",     // paraphrase of eligibility
  "Do I have to write any test to get admitted?",                 // paraphrase of entrance exam
  "By when should I submit my application?",                      // paraphrase of deadline
  "Can you tell me today's cricket score?",                       // out-of-scope -> fallback
];

async function main() {
  console.log("Loading model: Xenova/all-MiniLM-L6-v2 (public, no API key)...\n");
  const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  const faqs = await loadFaqDataset();
  console.log(`Embedding ${faqs.length} FAQ questions...\n`);
  const vectors = await buildIndex(extractor, faqs);

  const isDemo = process.argv.includes("--demo");
  const cliQuery = process.argv.slice(2).filter((a) => !a.startsWith("--")).join(" ");

  const queries = isDemo || !cliQuery ? DEMO_QUERIES : [cliQuery];

  for (const query of queries) {
    const result = await answerQuery(extractor, faqs, vectors, query);
    console.log(JSON.stringify(result, null, 2));
    console.log("---");
  }
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
