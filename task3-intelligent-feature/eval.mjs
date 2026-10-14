// Task 3 — Evaluation harness
//
// Runs a labeled test set through the assistant and reports accuracy,
// per-case pass/fail, and confidence-tier breakdown. The point is to show
// real measured behaviour — including genuine failures — not just curated
// demo queries that are guaranteed to work.

import { pipeline } from "@xenova/transformers";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Re-implement the minimal pieces needed for eval without importing
// assistant.mjs's module-level constants, so this file can also be read
// standalone as "here is exactly what's being measured".
import { cos_sim } from "@xenova/transformers";

const HIGH_CONFIDENCE = 0.55;
const LOW_CONFIDENCE = 0.35;

async function embed(extractor, text) {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// expected: the FAQ "question" string the query should match, or null if the
// query is genuinely out-of-scope and should fall back.
const TEST_CASES = [
  { query: "How much percentage do I need in 12th grade to get in?", expected: "What is the minimum eligibility criteria for B.Tech admission?" },
  { query: "Do I have to write any test to get admitted?", expected: "Is there an entrance exam required for admission?" },
  { query: "By when should I submit my application?", expected: "What is the last date to submit the application form?" },
  { query: "What papers do I need to bring for admission?", expected: "What documents are required for admission?" },
  { query: "Is financial aid or scholarship available?", expected: "Are there any scholarships available?" },
  { query: "How can I apply for an internship?", expected: "How do I apply for an internship under the program?" },
  { query: "Can I get my money back if I leave after joining?", expected: "What is the refund policy if I withdraw after admission?" },
  { query: "How long does the internship last?", expected: "What is the duration of the internship program?" },
  { query: "Is there a hostel for students from other cities?", expected: "Is hostel accommodation available for outstation students?" },
  { query: "What's today's weather like?", expected: null },
  { query: "Can you tell me today's cricket score?", expected: null },
  { query: "Write me a poem about admissions.", expected: null },
  // Deliberately hard / adversarial cases, included to document real
  // failure modes honestly rather than hide them:
  { query: "fee", expected: "What is the total course fee for B.Tech?" }, // single-word query, weak signal
  { query: "I already submitted my 12th marksheet, now what documents are left?", expected: "What documents are required for admission?" }, // long, multi-intent
  { query: "money", expected: "What is the total course fee for B.Tech?" }, // extremely underspecified, expected to fail
];

async function main() {
  const faqs = JSON.parse(await readFile(path.join(__dirname, "faq_dataset.json"), "utf-8"));
  console.log("Loading model: Xenova/all-MiniLM-L6-v2...\n");
  const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  const vectors = [];
  for (const faq of faqs) vectors.push(await embed(extractor, faq.question));

  let correct = 0;
  const rows = [];
  const confidenceCounts = { high: 0, medium: 0, low: 0 };

  for (const { query, expected } of TEST_CASES) {
    const qVec = await embed(extractor, query);
    const scored = vectors
      .map((v, i) => ({ i, score: cos_sim(qVec, v) }))
      .sort((a, b) => b.score - a.score);
    const best = scored[0];
    const confidence = best.score >= HIGH_CONFIDENCE ? "high" : best.score >= LOW_CONFIDENCE ? "medium" : "low";
    confidenceCounts[confidence]++;

    const predicted = confidence === "low" ? null : faqs[best.i].question;
    const pass = predicted === expected;
    if (pass) correct++;

    rows.push({
      query,
      expected: expected ?? "(no match expected)",
      predicted: predicted ?? "(no match)",
      similarity: Number(best.score.toFixed(4)),
      confidence,
      pass,
    });
  }

  console.log("Per-case results:\n");
  for (const r of rows) {
    console.log(`${r.pass ? "PASS" : "FAIL"} [${r.confidence}, sim=${r.similarity}]  "${r.query}"`);
    if (!r.pass) {
      console.log(`     expected:  ${r.expected}`);
      console.log(`     predicted: ${r.predicted}`);
    }
  }

  console.log("\nConfidence tier distribution:", confidenceCounts);
  console.log(`\nAccuracy: ${correct}/${TEST_CASES.length} (${((correct / TEST_CASES.length) * 100).toFixed(1)}%)`);
}

main().catch((err) => {
  console.error("Eval run failed:", err);
  process.exit(1);
});
