import { MAX_EVIDENCE_CHARS } from "./defaults.js";

const REVIEW_PATTERNS = [
  /\breview\b/i,
  /\bcustomer[s]?\b/i,
  /\brating[s]?\b/i,
  /\bstar[s]?\b/i,
  /\bverified\b/i,
  /\bpros?\b/i,
  /\bcons?\b/i
];

const TITLE_SELECTORS = [
  "[data-testid*='title' i]",
  "[class*='product-title' i]",
  "[class*='product__title' i]",
  "[id*='productTitle' i]",
  "h1"
];

const EVIDENCE_SELECTORS = [
  "main",
  "[role='main']",
  "[class*='product' i]",
  "[id*='product' i]",
  "[class*='review' i]",
  "[id*='review' i]",
  "article"
];

export function trimEvidenceText(text = "", maxChars = MAX_EVIDENCE_CHARS) {
  const reviewLines = extractReviewSnippetsFromText(text);
  const normalized = normalizeWhitespace(text);
  if (normalized.length <= maxChars) {
    return normalized;
  }

  const highSignal = reviewLines.join(" ");
  if (highSignal.length >= Math.min(40, maxChars)) {
    return highSignal.slice(0, maxChars).trim();
  }

  const headBudget = Math.floor(maxChars * 0.65);
  const tailBudget = maxChars - headBudget - 12;
  return `${normalized.slice(0, headBudget).trim()} ... ${normalized
    .slice(-tailBudget)
    .trim()}`.slice(0, maxChars);
}

export function extractReviewSnippetsFromText(text = "", limit = 8) {
  return text
    .split(/\n+/)
    .map((line) => normalizeWhitespace(line))
    .filter((line) => line.length >= 24)
    .filter((line) => REVIEW_PATTERNS.some((pattern) => pattern.test(line)))
    .slice(0, limit);
}

export function normalizeProductContext(context = {}) {
  const title = normalizeWhitespace(context.title) || "Untitled product page";
  const hostname = normalizeWhitespace(context.hostname);
  const url = normalizeWhitespace(context.url);
  const evidenceText = trimEvidenceText(context.evidenceText || "");
  const reviewSnippets = Array.isArray(context.reviewSnippets)
    ? context.reviewSnippets.map(normalizeWhitespace).filter(Boolean).slice(0, 8)
    : extractReviewSnippetsFromText(evidenceText);

  return {
    title,
    hostname,
    url,
    evidenceText,
    reviewSnippets,
    isSupported: title !== "Untitled product page" && evidenceText.length >= 160
  };
}

export function extractProductContextFromDocument(doc = document, locationLike = location) {
  const title =
    readMeta(doc, "og:title") ||
    readFirstMatchingText(doc, TITLE_SELECTORS) ||
    doc.title ||
    "";

  const chunks = [];
  const seen = new Set();
  for (const selector of EVIDENCE_SELECTORS) {
    for (const node of doc.querySelectorAll(selector)) {
      const text = node.innerText || node.textContent || "";
      const normalized = normalizeWhitespace(text);
      if (normalized.length < 40 || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      chunks.push(text);
    }
  }

  if (chunks.length === 0 && doc.body) {
    chunks.push(doc.body.innerText || doc.body.textContent || "");
  }

  const evidenceText = trimEvidenceText(chunks.join("\n"), MAX_EVIDENCE_CHARS);

  return normalizeProductContext({
    title,
    url: locationLike.href || "",
    hostname: locationLike.hostname || "",
    evidenceText,
    reviewSnippets: extractReviewSnippetsFromText(chunks.join("\n"))
  });
}

function readMeta(doc, property) {
  const node = doc.querySelector(`meta[property="${property}"], meta[name="${property}"]`);
  return normalizeWhitespace(node?.getAttribute("content") || "");
}

function readFirstMatchingText(doc, selectors) {
  for (const selector of selectors) {
    const node = doc.querySelector(selector);
    const text = normalizeWhitespace(node?.innerText || node?.textContent || "");
    if (text) {
      return text;
    }
  }
  return "";
}

function normalizeWhitespace(value = "") {
  return String(value).replace(/\s+/g, " ").trim();
}
