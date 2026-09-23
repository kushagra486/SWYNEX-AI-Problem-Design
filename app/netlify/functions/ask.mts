import type { Context, Config } from "@netlify/functions";

interface FaqEntry {
  question: string;
  answer: string;
  category: string;
}

const FAQ: FaqEntry[] = [
  { question: "What is the minimum eligibility criteria for B.Tech admission?", answer: "Candidates must have passed 10+2 with Physics, Chemistry, and Mathematics with a minimum aggregate of 45% (40% for reserved categories)", category: "eligibility" },
  { question: "Is there an entrance exam required for admission?", answer: "Yes, admission is generally through UPSEE/AKTU counselling or a valid JEE Main score, followed by institute-level counselling", category: "eligibility" },
  { question: "What is the last date to submit the application form?", answer: "The last date for form submission is typically announced on the official admission portal; check the notice board for the current cycle's deadline", category: "deadlines" },
  { question: "Can I apply after the deadline has passed?", answer: "Late applications are only accepted with a late fee if seats remain vacant, subject to management discretion", category: "deadlines" },
  { question: "What documents are required for admission?", answer: "You need your 10th and 12th mark sheets, transfer certificate, character certificate, category certificate (if applicable), and passport-size photographs", category: "documents" },
  { question: "Do I need to submit an income certificate?", answer: "An income certificate is required only if you are applying for a fee concession or scholarship under a reserved category", category: "documents" },
  { question: "What is the total course fee for B.Tech?", answer: "The fee structure varies by branch and is published on the official fee structure page each academic year", category: "fees" },
  { question: "Are there any scholarships available?", answer: "Yes, scholarships are available under state government schemes for eligible categories, and merit-based fee waivers for top-ranking students", category: "fees" },
  { question: "How do I apply for an internship under the program?", answer: "Fill out the online internship application form on the official portal and upload your resume and academic transcripts", category: "application_process" },
  { question: "What is the selection process for internships?", answer: "Selection is based on resume screening followed by a technical or HR interview round depending on the role", category: "application_process" },
  { question: "Can I defer my admission to the next academic year?", answer: "Deferral is allowed only in exceptional circumstances and requires written approval from the admission committee", category: "eligibility" },
  { question: "Is hostel accommodation available for outstation students?", answer: "Yes, hostel accommodation is available on a first-come-first-served basis; apply separately through the hostel admission form", category: "documents" },
  { question: "What is the refund policy if I withdraw after admission?", answer: "Refunds follow the AICTE-prescribed fee refund policy based on the date of withdrawal request", category: "fees" },
  { question: "Who do I contact for admission-related queries?", answer: "You can contact the admission cell via the official email or phone number listed on the institute's contact page", category: "application_process" },
  { question: "What is the duration of the internship program?", answer: "Most internship programs run for 6 to 12 weeks depending on the track selected", category: "application_process" },
];

const STOPWORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "i", "you", "he", "she", "it",
  "we", "they", "my", "your", "his", "her", "its", "our", "their", "need", "to", "for", "of", "in", "on", "at", "by", "with",
  "what", "how", "can", "could", "should", "would", "will", "this", "that", "these", "those", "and", "or", "if", "be", "am",
  "has", "have", "had", "not", "no", "yes", "please", "about", "there", "get", "also", "me", "us",
]);

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t && !STOPWORDS.has(t));
}

interface Index {
  vocab: Map<string, number>;
  vectors: Float64Array[];
  df: number[];
  docCount: number;
}

function buildIndex(docs: string[]): Index {
  const vocab = new Map<string, number>();
  docs.forEach((d) => tokenize(d).forEach((t) => { if (!vocab.has(t)) vocab.set(t, vocab.size); }));

  const df = new Array(vocab.size).fill(0);
  const tfs = docs.map((doc) => {
    const tokens = tokenize(doc);
    const tf = new Map<string, number>();
    tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    return tf;
  });
  tfs.forEach((tf) => { for (const t of tf.keys()) df[vocab.get(t)!]++; });

  const N = docs.length;
  const vectors = tfs.map((tf) => {
    const vec = new Float64Array(vocab.size);
    for (const [t, c] of tf.entries()) {
      const idx = vocab.get(t)!;
      const idf = Math.log((N + 1) / (df[idx] + 1)) + 1;
      vec[idx] = c * idf;
    }
    return vec;
  });

  return { vocab, vectors, df, docCount: N };
}

function vectorizeQuery(text: string, index: Index): Float64Array {
  const tokens = tokenize(text);
  const tf = new Map<string, number>();
  tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
  const vec = new Float64Array(index.vocab.size);
  for (const [t, c] of tf.entries()) {
    if (!index.vocab.has(t)) continue;
    const idx = index.vocab.get(t)!;
    const idf = Math.log((index.docCount + 1) / (index.df[idx] + 1)) + 1;
    vec[idx] = c * idf;
  }
  return vec;
}

function cosine(a: Float64Array, b: Float64Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const THRESHOLD = 0.12;
const INDEX = buildIndex(FAQ.map((f) => f.question));

export default async (req: Request, context: Context) => {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "content-type": "application/json" },
    });
  }

  const body = await req.json().catch(() => null);
  const query = body && typeof body.query === "string" ? body.query.trim() : "";
  if (!query) {
    return new Response(JSON.stringify({ error: "Missing 'query' field" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  const qvec = vectorizeQuery(query, INDEX);
  let best = -1, bestScore = -1;
  INDEX.vectors.forEach((v, i) => {
    const s = cosine(qvec, v);
    if (s > bestScore) { bestScore = s; best = i; }
  });

  if (bestScore < THRESHOLD) {
    return new Response(
      JSON.stringify({
        matched: false,
        answer: "I don't have this information — please contact the admission cell.",
        similarity: bestScore,
      }),
      { headers: { "content-type": "application/json" } }
    );
  }

  const match = FAQ[best];
  return new Response(
    JSON.stringify({
      matched: true,
      answer: match.answer,
      category: match.category,
      similarity: bestScore,
    }),
    { headers: { "content-type": "application/json" } }
  );
};

export const config: Config = {
  path: "/api/ask",
};
