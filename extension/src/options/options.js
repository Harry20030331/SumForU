const STORAGE_KEYS = {
  backendUrl: "sumforu.backendUrl",
  persona: "sumforu.persona"
};
const DEFAULT_BACKEND_URL = "http://127.0.0.1:8788";
const DEFAULT_PERSONA =
  "A practical shopper who wants good long-term value, low hassle, and clear tradeoffs before buying.";

const form = document.querySelector("#settingsForm");
const personaInput = document.querySelector("#personaInput");
const backendUrlInput = document.querySelector("#backendUrlInput");
const statusText = document.querySelector("#statusText");

document.addEventListener("DOMContentLoaded", loadSettings);
form.addEventListener("submit", saveSettings);

async function loadSettings() {
  const settings = await chrome.storage.local.get([
    STORAGE_KEYS.backendUrl,
    STORAGE_KEYS.persona
  ]);

  backendUrlInput.value = settings[STORAGE_KEYS.backendUrl] || DEFAULT_BACKEND_URL;
  personaInput.value = settings[STORAGE_KEYS.persona] || DEFAULT_PERSONA;
}

async function saveSettings(event) {
  event.preventDefault();
  await chrome.storage.local.set({
    [STORAGE_KEYS.backendUrl]: normalizeBackendUrl(backendUrlInput.value.trim() || DEFAULT_BACKEND_URL),
    [STORAGE_KEYS.persona]: personaInput.value.trim() || DEFAULT_PERSONA
  });
  statusText.textContent = "Settings saved locally in Chrome.";
}

function normalizeBackendUrl(value) {
  return value.replace(/\/+$/, "");
}
