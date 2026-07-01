# Chrome Extension Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a demonstrable Chrome extension for Summary for You that generates persona-aware product buying summaries with OpenAI.

**Architecture:** Add an isolated `extension/` app with MV3 files, shared modules, tests, and local demo pages. Keep generation in the service worker, extraction in the content script, and rendering in popup/options pages.

**Tech Stack:** Chrome Manifest V3, vanilla HTML/CSS/JavaScript modules, Node's built-in test runner for pure logic tests.

---

## File Structure

- `extension/manifest.json`: Chrome extension manifest.
- `extension/src/content/contentScript.js`: Extracts product context from product pages.
- `extension/src/background/serviceWorker.js`: Handles OpenAI generation and storage lookup.
- `extension/src/shared/extraction.js`: Pure extraction and trimming helpers.
- `extension/src/shared/openaiClient.js`: Prompt construction, API request, JSON parsing, schema normalization.
- `extension/src/shared/defaults.js`: Default persona, model, recommendation labels, storage keys.
- `extension/src/popup/popup.html`: Popup UI shell.
- `extension/src/popup/popup.css`: Popup styling.
- `extension/src/popup/popup.js`: Popup state and Chrome messaging.
- `extension/src/options/options.html`: API key/settings UI.
- `extension/src/options/options.css`: Settings styling.
- `extension/src/options/options.js`: Save/load settings.
- `extension/tests/*.test.mjs`: Pure module tests.
- `extension/demo-pages/*.html`: Reproducible demo pages.
- `extension/README.md`: Loading, testing, privacy, and Chrome Web Store caveats.
- Root `README.md`: Short pointer to the extension demo.
- Root `package.json`: Test script for extension pure logic tests.

## Tasks

- [ ] Create extension scaffold and manifest.
- [ ] Add pure extraction helpers with tests, verifying conservative visible text trimming.
- [ ] Add OpenAI prompt/client helpers with tests, verifying JSON recovery and schema normalization.
- [ ] Build popup UI and states.
- [ ] Build options page for local API key and model settings.
- [ ] Wire content script, popup, and service worker messages.
- [ ] Add demo product pages.
- [ ] Document setup and caveats.
- [ ] Run automated tests and static extension sanity checks.
