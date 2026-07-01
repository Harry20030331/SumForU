import { DEFAULT_BACKEND_URL, DEFAULT_PERSONA, STORAGE_KEYS } from "../shared/defaults.js";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "SUMFORU_GENERATE") {
    return false;
  }

  handleGenerate(message.payload)
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});

async function handleGenerate(payload = {}) {
  const settings = await chrome.storage.local.get([
    STORAGE_KEYS.backendUrl,
    STORAGE_KEYS.installId,
    STORAGE_KEYS.persona
  ]);

  const backendUrl = normalizeBackendUrl(settings[STORAGE_KEYS.backendUrl] || DEFAULT_BACKEND_URL);
  const persona = payload.persona || settings[STORAGE_KEYS.persona] || DEFAULT_PERSONA;
  const installId = await getInstallId(settings[STORAGE_KEYS.installId]);

  if (!payload.context?.isSupported) {
    throw new Error("This page does not look like a supported product page yet.");
  }

  const response = await fetch(`${backendUrl}/api/summary`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-SumForU-Install-Id": installId
    },
    body: JSON.stringify({
      persona,
      context: payload.context
    })
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Summary service failed (${response.status}).`);
  }

  return body.result;
}

async function getInstallId(existing) {
  if (existing) {
    return existing;
  }
  const installId = crypto.randomUUID();
  await chrome.storage.local.set({ [STORAGE_KEYS.installId]: installId });
  return installId;
}

function normalizeBackendUrl(value) {
  return String(value || DEFAULT_BACKEND_URL).replace(/\/+$/, "");
}
