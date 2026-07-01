const MAX_EVIDENCE_CHARS = 7000;
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

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "SUMFORU_GET_CONTEXT") {
    return false;
  }

  sendResponse({ ok: true, context: extractProductContext() });
  return false;
});

function extractProductContext() {
  const title =
    readMeta("og:title") || readFirstMatchingText(TITLE_SELECTORS) || document.title || "";
  const chunks = [];
  const seen = new Set();

  for (const selector of EVIDENCE_SELECTORS) {
    for (const node of document.querySelectorAll(selector)) {
      const text = node.innerText || node.textContent || "";
      const normalized = normalizeWhitespace(text);
      if (normalized.length < 40 || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      chunks.push(text);
    }
  }

  if (chunks.length === 0 && document.body) {
    chunks.push(document.body.innerText || document.body.textContent || "");
  }

  const evidenceSource = chunks.join("\n");
  const evidenceText = trimEvidenceText(evidenceSource, MAX_EVIDENCE_CHARS);
  const normalizedTitle = normalizeWhitespace(title) || "Untitled product page";

  return {
    title: normalizedTitle,
    url: location.href,
    hostname: location.hostname,
    evidenceText,
    reviewSnippets: extractReviewSnippetsFromText(evidenceSource),
    isSupported: normalizedTitle !== "Untitled product page" && evidenceText.length >= 160
  };
}

function trimEvidenceText(text, maxChars) {
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

function extractReviewSnippetsFromText(text, limit = 8) {
  return String(text || "")
    .split(/\n+/)
    .map((line) => normalizeWhitespace(line))
    .filter((line) => line.length >= 24)
    .filter((line) => REVIEW_PATTERNS.some((pattern) => pattern.test(line)))
    .slice(0, limit);
}

function readMeta(property) {
  const node = document.querySelector(`meta[property="${property}"], meta[name="${property}"]`);
  return normalizeWhitespace(node?.getAttribute("content") || "");
}

function readFirstMatchingText(selectors) {
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    const text = normalizeWhitespace(node?.innerText || node?.textContent || "");
    if (text) {
      return text;
    }
  }
  return "";
}

function normalizeWhitespace(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}
