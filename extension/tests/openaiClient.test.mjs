import test from "node:test";
import assert from "node:assert/strict";

import {
  buildSummaryPrompt,
  normalizeSummaryResult,
  parseSummaryJson
} from "../src/shared/openaiClient.js";

test("buildSummaryPrompt asks for grounded JSON personalized to persona", () => {
  const prompt = buildSummaryPrompt({
    persona: "Budget-conscious student who cares about durability.",
    context: {
      title: "Trail Mug",
      hostname: "demo.local",
      url: "https://demo.local/mug.html",
      evidenceText: "Double-wall mug. Reviews mention leaks when tossed in backpacks."
    }
  });

  assert.match(prompt, /Budget-conscious student/);
  assert.match(prompt, /Do not invent/);
  assert.match(prompt, /suitabilityScore/);
});

test("parseSummaryJson recovers JSON wrapped in prose", () => {
  const result = parseSummaryJson(`
    Here is the result:
    {"productSummary":"Compact","personaFit":"Good","strengths":["Small"],"concerns":["Price"],"suitabilityScore":7,"recommendation":"Good fit","grounding":"Based on review text."}
  `);

  assert.equal(result.recommendation, "Good fit");
  assert.equal(result.suitabilityScore, 7);
});

test("normalizeSummaryResult clamps score and fills arrays", () => {
  const result = normalizeSummaryResult({
    productSummary: "Useful",
    suitabilityScore: 42,
    recommendation: "surprising",
    strengths: "Portable",
    concerns: null
  });

  assert.equal(result.suitabilityScore, 10);
  assert.equal(result.recommendation, "Mixed");
  assert.deepEqual(result.strengths, ["Portable"]);
  assert.deepEqual(result.concerns, []);
});
