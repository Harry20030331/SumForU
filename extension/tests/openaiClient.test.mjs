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
  assert.match(prompt, /SumForU/);
  assert.match(prompt, /Keep the whole response compact/);
  assert.match(prompt, /2 or 3/);
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
    productSummary: "Useful ".repeat(40),
    personaFit: "Good ".repeat(60),
    suitabilityScore: 42,
    recommendation: "surprising",
    strengths: [
      "Portable ".repeat(30),
      "Durable",
      "Easy",
      "Extra item"
    ],
    concerns: ["Price ".repeat(30), "Warranty", "Availability", "Extra concern"],
    grounding: "Based on review text ".repeat(20)
  });

  assert.equal(result.suitabilityScore, 10);
  assert.equal(result.recommendation, "Mixed");
  assert.equal(result.strengths.length, 3);
  assert.equal(result.concerns.length, 3);
  assert.equal(result.strengths[0].split(/\s+/).length <= 16, true);
  assert.equal(result.concerns[0].split(/\s+/).length <= 16, true);
  assert.equal(result.productSummary.split(/\s+/).length <= 28, true);
  assert.equal(result.personaFit.split(/\s+/).length <= 36, true);
  assert.equal(result.grounding.split(/\s+/).length <= 22, true);
});
