import test from "node:test";
import assert from "node:assert/strict";

import {
  extractReviewSnippetsFromText,
  normalizeProductContext,
  trimEvidenceText
} from "../src/shared/extraction.js";

test("trimEvidenceText keeps high-signal text within budget", () => {
  const text = [
    "Noise ".repeat(80),
    "Customer reviews say the battery lasts all day and setup is simple.",
    "More filler ".repeat(80)
  ].join("\n");

  const result = trimEvidenceText(text, 120);

  assert.ok(result.length <= 120);
  assert.match(result, /Customer reviews/);
});

test("extractReviewSnippetsFromText finds review-like lines", () => {
  const result = extractReviewSnippetsFromText(`
    Product details: brushed steel kettle.
    Review: heats quickly but the handle gets warm.
    Customers mention it is easy to clean.
  `);

  assert.deepEqual(result, [
    "Review: heats quickly but the handle gets warm.",
    "Customers mention it is easy to clean."
  ]);
});

test("normalizeProductContext produces a usable unsupported-page flag", () => {
  const result = normalizeProductContext({
    title: "   ",
    url: "https://example.com/listing",
    hostname: "example.com",
    evidenceText: "short"
  });

  assert.equal(result.isSupported, false);
  assert.equal(result.title, "Untitled product page");
  assert.equal(result.hostname, "example.com");
});
