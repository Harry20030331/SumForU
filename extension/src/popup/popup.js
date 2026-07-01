const STORAGE_KEYS = {
  backendUrl: "sumforu.backendUrl",
  installId: "sumforu.installId",
  persona: "sumforu.persona"
};
const DEFAULT_PERSONA =
  "A practical shopper who wants good long-term value, low hassle, and clear tradeoffs before buying.";

const state = {
  tabId: null,
  context: null
};

const els = {
  settingsButton: document.querySelector("#settingsButton"),
  refreshButton: document.querySelector("#refreshButton"),
  generateButton: document.querySelector("#generateButton"),
  productTitle: document.querySelector("#productTitle"),
  siteLabel: document.querySelector("#siteLabel"),
  supportText: document.querySelector("#supportText"),
  personaInput: document.querySelector("#personaInput"),
  statusPanel: document.querySelector("#statusPanel"),
  resultPanel: document.querySelector("#resultPanel"),
  recommendationLabel: document.querySelector("#recommendationLabel"),
  scoreValue: document.querySelector("#scoreValue"),
  productSummary: document.querySelector("#productSummary"),
  personaFit: document.querySelector("#personaFit"),
  strengthsList: document.querySelector("#strengthsList"),
  concernsList: document.querySelector("#concernsList"),
  groundingText: document.querySelector("#groundingText")
};

document.addEventListener("DOMContentLoaded", init);
els.settingsButton.addEventListener("click", () => chrome.runtime.openOptionsPage());
els.refreshButton.addEventListener("click", loadContext);
els.generateButton.addEventListener("click", generateSummary);
els.personaInput.addEventListener("change", () => {
  chrome.storage.local.set({ [STORAGE_KEYS.persona]: els.personaInput.value.trim() });
});

async function init() {
  const settings = await chrome.storage.local.get([STORAGE_KEYS.persona]);
  els.personaInput.value = settings[STORAGE_KEYS.persona] || DEFAULT_PERSONA;
  await loadContext();
}

async function loadContext() {
  setStatus("Reading this tab...", "info");
  hideResult();
  els.generateButton.disabled = true;

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    state.tabId = tab?.id;
    if (!state.tabId) {
      throw new Error("No active tab found.");
    }

    const response = await chrome.tabs.sendMessage(state.tabId, { type: "SUMFORU_GET_CONTEXT" });
    if (!response?.ok) {
      throw new Error(response?.error || "Could not read this page.");
    }

    state.context = response.context;
    renderContext(response.context);
    clearStatus();
  } catch (error) {
    state.context = null;
    renderContext(null);
    setStatus(
      "I could not read this page. Reload the page after installing the extension, then try again.",
      "error"
    );
  }
}

async function generateSummary() {
  if (!state.context?.isSupported) {
    setStatus("This page does not have enough product evidence for a useful summary.", "error");
    return;
  }

  const persona = els.personaInput.value.trim() || DEFAULT_PERSONA;
  await chrome.storage.local.set({ [STORAGE_KEYS.persona]: persona });

  els.generateButton.disabled = true;
  setStatus("Generating personalized recommendation...", "info");
  hideResult();

  try {
    const response = await chrome.runtime.sendMessage({
      type: "SUMFORU_GENERATE",
      payload: { context: state.context, persona }
    });
    if (!response?.ok) {
      throw new Error(response?.error || "Generation failed.");
    }
    renderResult(response.result);
    clearStatus();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    els.generateButton.disabled = false;
  }
}

function renderContext(context) {
  if (!context) {
    els.siteLabel.textContent = "No readable page";
    els.productTitle.textContent = "Product page";
    els.supportText.textContent = "Open a product page and refresh.";
    els.generateButton.disabled = true;
    return;
  }

  els.siteLabel.textContent = context.hostname || "Current tab";
  els.productTitle.textContent = context.title;
  els.supportText.textContent = context.isSupported
    ? `${context.evidenceText.length.toLocaleString()} characters of page evidence detected.`
    : "This page has limited product evidence; try a product detail page.";
  els.generateButton.disabled = !context.isSupported;
}

function renderResult(result) {
  els.recommendationLabel.textContent = result.recommendation;
  els.scoreValue.textContent = result.suitabilityScore;
  els.productSummary.textContent = result.productSummary;
  els.personaFit.textContent = result.personaFit;
  renderList(els.strengthsList, result.strengths);
  renderList(els.concernsList, result.concerns);
  els.groundingText.textContent = result.grounding;
  els.resultPanel.classList.remove("hidden");
}

function renderList(node, items) {
  node.replaceChildren(
    ...items.map((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      return li;
    })
  );
}

function setStatus(message, tone) {
  els.statusPanel.textContent = message;
  els.statusPanel.dataset.tone = tone;
  els.statusPanel.classList.remove("hidden");
}

function clearStatus() {
  els.statusPanel.textContent = "";
  els.statusPanel.classList.add("hidden");
}

function hideResult() {
  els.resultPanel.classList.add("hidden");
}
