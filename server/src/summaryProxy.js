import {
  buildSummaryPrompt,
  normalizeSummaryResult,
  parseSummaryJson
} from "../../extension/src/shared/openaiClient.js";

const DEFAULT_MODEL = "gpt-5-mini";
const MAX_EVIDENCE_CHARS = 9000;
const MAX_PERSONA_CHARS = 1200;
const MIN_EVIDENCE_CHARS = 120;
const ALLOWED_RECOMMENDATIONS = new Set([
  "Strong fit",
  "Good fit",
  "Mixed",
  "Not recommended"
]);

export function createSummaryHandler({ env = process.env, fetchImpl = fetch } = {}) {
  const limiter = createDailyLimiter(Number(env.SUMFORU_DAILY_LIMIT || 30));

  return async function summaryHandler(request) {
    try {
      if (request.method !== "POST") {
        return json({ error: "Method not allowed" }, 405);
      }

      const apiKey = env.OPENAI_API_KEY;
      if (!apiKey) {
        return json({ error: "Server is missing OPENAI_API_KEY." }, 500);
      }

      const installId = cleanText(request.headers.get("x-sumforu-install-id") || "anonymous");
      if (!limiter.allow(installId)) {
        return json({ error: "Daily demo limit reached. Try again tomorrow." }, 429);
      }

      const payload = await request.json();
      const input = validateSummaryRequest(payload, {
        model: env.OPENAI_MODEL || DEFAULT_MODEL
      });
      const result = await callOpenAI({ input, apiKey, fetchImpl });

      return json({ result });
    } catch (error) {
      const status = /persona|context|evidence|JSON|url/i.test(error.message) ? 400 : 502;
      return json({ error: error.message }, status);
    }
  };
}

function createDailyLimiter(limit) {
  const counts = new Map();

  return {
    allow(id) {
      const day = new Date().toISOString().slice(0, 10);
      const key = `${day}:${id}`;
      const count = counts.get(key) || 0;
      if (count >= limit) {
        return false;
      }
      counts.set(key, count + 1);
      return true;
    }
  };
}

export function validateSummaryRequest(payload = {}, { model = DEFAULT_MODEL } = {}) {
  const persona = cleanText(payload.persona).slice(0, MAX_PERSONA_CHARS);
  if (persona.length < 12) {
    throw new Error("A shopper persona is required.");
  }

  const context = payload.context || {};
  const title = cleanText(context.title) || "Untitled product page";
  const hostname = cleanText(context.hostname);
  const url = cleanText(context.url);
  const evidenceText = cleanText(context.evidenceText).slice(0, MAX_EVIDENCE_CHARS);

  if (!url || !/^https?:\/\//i.test(url)) {
    throw new Error("A valid product page URL is required.");
  }

  if (evidenceText.length < MIN_EVIDENCE_CHARS) {
    throw new Error("The page does not have enough product evidence to summarize.");
  }

  return {
    model,
    persona,
    context: {
      title,
      hostname,
      url,
      evidenceText,
      reviewSnippets: Array.isArray(context.reviewSnippets)
        ? context.reviewSnippets.map(cleanText).filter(Boolean).slice(0, 8)
        : [],
      isSupported: Boolean(context.isSupported)
    }
  };
}

async function callOpenAI({ input, apiKey, fetchImpl }) {
  const body = {
    model: input.model,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "Return concise, grounded buying guidance as JSON. Never include markdown or unsupported claims."
      },
      {
        role: "user",
        content: buildSummaryPrompt(input)
      }
    ]
  };

  if (!input.model.startsWith("gpt-5")) {
    body.temperature = 0.2;
  }

  const response = await fetchImpl("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenAI request failed (${response.status}): ${text.slice(0, 220)}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  const normalized = normalizeSummaryResult(parseSummaryJson(content));
  normalized.recommendation = ALLOWED_RECOMMENDATIONS.has(normalized.recommendation)
    ? normalized.recommendation
    : "Mixed";
  return normalized;
}

function cleanText(value = "") {
  return String(value).replace(/\s+/g, " ").trim();
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  });
}
