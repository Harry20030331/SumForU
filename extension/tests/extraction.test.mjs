import test from "node:test";
import assert from "node:assert/strict";

import {
  extractProductContextFromDocument,
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

test("trimEvidenceText preserves context when review signals are thin", () => {
  const text = [
    "Apple AirPods Pro 2 product title and price $239.99 ".repeat(80),
    "Rating 4.8 out of 5 stars with 32063 reviews.",
    "Shipping and setup details for Apple users ".repeat(80)
  ].join("\n");

  const result = trimEvidenceText(text, 600);

  assert.ok(result.length <= 600);
  assert.match(result, /Apple AirPods Pro 2 product title/);
  assert.match(result, /Shipping and setup details/);
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

test("extractProductContextFromDocument prefers h1 product title over generic title labels", () => {
  const doc = fakeDocument({
    title: "Fallback browser title",
    nodes: {
      h1: ["Apple - AirPods Pro 2, Wireless Active Noise Cancelling Earbuds - White"],
      "[data-testid*='title' i]": ["Pickup"],
      main: [
        "Apple - AirPods Pro 2, Wireless Active Noise Cancelling Earbuds - White\nRating 4.8 out of 5 stars with 32063 reviews\n$239.99"
      ]
    }
  });

  const result = extractProductContextFromDocument(doc, {
    href: "https://www.bestbuy.com/product/airpods",
    hostname: "www.bestbuy.com"
  });

  assert.equal(
    result.title,
    "Apple - AirPods Pro 2, Wireless Active Noise Cancelling Earbuds - White"
  );
});

function fakeDocument({ title, nodes }) {
  return {
    title,
    body: { innerText: "", textContent: "" },
    querySelector(selector) {
      const value = nodes[selector]?.[0];
      return value ? { innerText: value, textContent: value, getAttribute: () => "" } : null;
    },
    querySelectorAll(selector) {
      return (nodes[selector] || []).map((value) => ({
        innerText: value,
        textContent: value
      }));
    }
  };
}
