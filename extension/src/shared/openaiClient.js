import { DEFAULT_MODEL, RECOMMENDATION_LABELS } from "./defaults.js";

export function buildSummaryPrompt({ persona, context }) {
  return `You are Summary for You, a shopping assistant that creates grounded, persona-aware product buying summaries.

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
- Return only valid JSON with this exact shape:
{
  "productSummary": "one sentence",
  "personaFit": "two to three sentences",
  "strengths": ["3 to 5 personalized strengths"],
  "concerns": ["3 to 5 personalized concerns"],
  "suitabilityScore": 1,
  "recommendation": "Strong fit | Good fit | Mixed | Not recommended",
  "grounding": "short explanation of what evidence the recommendation relies on"
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
    productSummary: asString(raw.productSummary, "No product summary returned."),
    personaFit: asString(raw.personaFit, "The available evidence is too thin to assess fit."),
    strengths: asList(raw.strengths).slice(0, 5),
    concerns: asList(raw.concerns).slice(0, 5),
    suitabilityScore: Number.isFinite(score) ? Math.min(10, Math.max(1, Math.round(score))) : 5,
    recommendation,
    grounding: asString(raw.grounding, "Grounding was not provided by the model.")
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
