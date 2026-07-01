import { DEFAULT_MODEL, RECOMMENDATION_LABELS } from "./defaults.js";

export function buildSummaryPrompt({ persona, context }) {
  return `You are Sum for You, a shopping assistant that creates grounded, persona-aware product buying summaries.

Shopper persona:
${persona}

Product page:
- Title: ${context.title}
- Site: ${context.hostname}
- URL: ${context.url}

Extracted page and review evidence:
${context.evidenceText}

Instructions:
- Treat the extracted text as imperfect evidence.
- Do not invent product claims, specifications, discounts, review counts, or guarantees not present in the evidence.
- Personalize the tradeoffs to the shopper persona.
- If evidence is thin or mixed, say so.
- Keep the whole response compact for a narrow Chrome extension popup.
- Prefer the highest-signal review patterns; do not over-explain.
- Return only valid JSON with this exact shape:
{
  "productSummary": "one short sentence, 22 words or fewer",
  "personaFit": "one short sentence, 30 words or fewer",
  "strengths": ["2 or 3 concise strengths, each 12 words or fewer"],
  "concerns": ["2 or 3 concise concerns, each 12 words or fewer"],
  "suitabilityScore": 1,
  "recommendation": "Strong fit | Good fit | Mixed | Not recommended",
  "grounding": "one evidence note, 18 words or fewer"
}`;
}

export async function generateSummaryWithOpenAI({
  apiKey,
  model = DEFAULT_MODEL,
  persona,
  context,
  fetchImpl = fetch
}) {
  if (!apiKey) {
    throw new Error("Missing OpenAI API key. Configure OPENAI_API_KEY on the backend.");
  }

  const response = await fetchImpl("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "Return concise, grounded product buying guidance as JSON. Never include markdown."
        },
        {
          role: "user",
          content: buildSummaryPrompt({ persona, context })
        }
      ]
    })
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenAI request failed (${response.status}): ${text.slice(0, 240)}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  return normalizeSummaryResult(parseSummaryJson(content));
}

export function parseSummaryJson(text = "") {
  const source = String(text).trim();
  try {
    return JSON.parse(source);
  } catch {
    const start = source.indexOf("{");
    const end = source.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) {
      throw new Error("The model did not return JSON.");
    }
    return JSON.parse(source.slice(start, end + 1));
  }
}

export function normalizeSummaryResult(raw = {}) {
  const score = Number(raw.suitabilityScore);
  const recommendation = RECOMMENDATION_LABELS.includes(raw.recommendation)
    ? raw.recommendation
    : "Mixed";

  return {
    productSummary: truncateWords(asString(raw.productSummary, "No product summary returned."), 28),
    personaFit: truncateWords(
      asString(raw.personaFit, "The available evidence is too thin to assess fit."),
      36
    ),
    strengths: asList(raw.strengths).slice(0, 3).map((item) => truncateWords(item, 16)),
    concerns: asList(raw.concerns).slice(0, 3).map((item) => truncateWords(item, 16)),
    suitabilityScore: Number.isFinite(score) ? Math.min(10, Math.max(1, Math.round(score))) : 5,
    recommendation,
    grounding: truncateWords(asString(raw.grounding, "Grounding was not provided by the model."), 22)
  };
}

function asString(value, fallback) {
  const text = String(value || "").trim();
  return text || fallback;
}

function asList(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) {
    return [value.trim()];
  }
  return [];
}

function truncateWords(value, maxWords) {
  const words = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) {
    return words.join(" ");
  }
  return `${words.slice(0, maxWords).join(" ")}...`;
}
