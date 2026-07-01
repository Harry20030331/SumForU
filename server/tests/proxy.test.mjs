import test from "node:test";
import assert from "node:assert/strict";

import { createServer } from "../src/app.js";
import { createSummaryHandler, validateSummaryRequest } from "../src/summaryProxy.js";

test("validateSummaryRequest rejects missing persona", () => {
  assert.throws(
    () =>
      validateSummaryRequest({
        context: {
          title: "Hotel",
          hostname: "hotels.com",
          url: "https://www.hotels.com/example",
          evidenceText: "A useful hotel page with review evidence.".repeat(8),
          isSupported: true
        }
      }),
    /persona/i
  );
});

test("validateSummaryRequest trims long evidence for cost control", () => {
  const result = validateSummaryRequest({
    persona: "A traveler who wants quiet rooms and walkable neighborhoods.",
    context: {
      title: "Innside by Melia New York Nomad",
      hostname: "www.hotels.com",
      url: "https://www.hotels.com/example",
      evidenceText: "Guest review: quiet room and good location. ".repeat(500),
      isSupported: true
    }
  });

  assert.equal(result.context.evidenceText.length <= 5000, true);
  assert.equal(result.model, "gpt-5-mini");
});

test("summary handler calls OpenAI with server-side key", async () => {
  const calls = [];
  const handler = createSummaryHandler({
    env: { OPENAI_API_KEY: "server-secret", OPENAI_MODEL: "gpt-5-mini" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  productSummary: "Central hotel with mixed tradeoffs.",
                  personaFit: "Good for a traveler who values location.",
                  strengths: ["Walkable location"],
                  concerns: ["Price is high"],
                  suitabilityScore: 7,
                  recommendation: "Good fit",
                  grounding: "Based on location and review evidence."
                })
              }
            }
          ]
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
  });

  const response = await handler(
    new Request("https://sumforu.test/api/summary", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        persona: "A traveler who wants quiet rooms and walkable neighborhoods.",
        context: {
          title: "Innside by Melia New York Nomad",
          hostname: "www.hotels.com",
          url: "https://www.hotels.com/example",
          evidenceText: "Guest review: quiet room and good location. ".repeat(20),
          isSupported: true
        }
      })
    })
  );

  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.result.recommendation, "Good fit");
  assert.equal(calls[0].url, "https://api.openai.com/v1/chat/completions");
  assert.match(calls[0].options.headers.Authorization, /server-secret/);
  assert.match(calls[0].options.body, /gpt-5-mini/);
  assert.equal("temperature" in JSON.parse(calls[0].options.body), false);
});

test("summary handler rate limits an anonymous install id", async () => {
  const handler = createSummaryHandler({
    env: { OPENAI_API_KEY: "server-secret", SUMFORU_DAILY_LIMIT: "1" },
    fetchImpl: async () =>
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  productSummary: "Useful",
                  personaFit: "Useful",
                  strengths: [],
                  concerns: [],
                  suitabilityScore: 5,
                  recommendation: "Mixed",
                  grounding: "Evidence."
                })
              }
            }
          ]
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
  });
  const body = JSON.stringify({
    persona: "A traveler who wants quiet rooms and walkable neighborhoods.",
    context: {
      title: "Hotel",
      hostname: "hotels.com",
      url: "https://www.hotels.com/example",
      evidenceText: "Guest review: quiet room and good location. ".repeat(20),
      isSupported: true
    }
  });

  const first = await handler(
    new Request("https://sumforu.test/api/summary", {
      method: "POST",
      headers: { "content-type": "application/json", "x-sumforu-install-id": "install-1" },
      body
    })
  );
  const second = await handler(
    new Request("https://sumforu.test/api/summary", {
      method: "POST",
      headers: { "content-type": "application/json", "x-sumforu-install-id": "install-1" },
      body
    })
  );

  assert.equal(first.status, 200);
  assert.equal(second.status, 429);
});

test("app responds to health checks", async () => {
  const app = createServer({ env: {}, fetchImpl: fetch });
  const response = await app.fetch(new Request("https://sumforu.test/healthz"));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true });
});
