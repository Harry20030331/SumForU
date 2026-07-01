# Summary for You Chrome Extension Demo Design

## Goal

Build a local-demo Chrome extension that turns the existing SumForU product idea into a visible browser experience. On a product page, the user can open the extension, review detected product context, choose a shopper persona, and generate a grounded, persona-aware buying summary with OpenAI.

## Scope

The demo is intentionally separate from the research pipeline. It does not use Tinker checkpoints, SVM artifacts, or dataset training scripts. It adds an isolated browser extension under `extension/` plus demo fixtures and README instructions.

## Architecture

- Manifest V3 extension with a popup, options page, content script, and service worker.
- Content script extracts visible product evidence from the active tab: title, URL, hostname, high-signal text, and obvious review blocks.
- Popup renders the primary workflow: detected product, editable persona, generate action, loading/error states, and structured results.
- Options page stores the tester's OpenAI API key and model choice in `chrome.storage.local`.
- Service worker owns the OpenAI request so popup code does not directly manage network generation logic.
- Shared JavaScript modules provide schema validation, prompt construction, extraction trimming, and deterministic demo helpers.

## User Experience

The popup should feel like a compact shopping advisor, not a research dashboard. It should show the product identity first, then persona controls, then a clear recommendation result: one-line summary, fit summary, strengths, concerns, suitability score, recommendation label, and grounding note.

The interface must handle:

- No API key configured.
- Unsupported or low-signal page.
- Loading.
- OpenAI/API failure.
- Malformed model JSON.
- Successful result.

## OpenAI Behavior

The prompt treats page text as imperfect extracted evidence and instructs the model not to invent unsupported claims. The model must return JSON with these fields:

- `productSummary`
- `personaFit`
- `strengths`
- `concerns`
- `suitabilityScore`
- `recommendation`
- `grounding`

The default model is a small OpenAI model suitable for local demo latency and cost. The API key is never committed and is only stored locally in Chrome storage for the tester's browser profile.

## Demo Fixtures

Add three saved HTML pages under `extension/demo-pages/`:

- Electronics product.
- Home/kitchen or beauty product.
- Ambiguous product where persona changes the recommendation.

Each page should contain product details and review-like evidence so the extension can be tested without external site variability.

## Verification

The implementation should include automated tests for pure logic such as extraction trimming, prompt/schema handling, and fallback parsing. Manual verification should cover loading the unpacked extension in Chrome, opening demo pages, saving a key, generating a result, and checking error states.
