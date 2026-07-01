export const STORAGE_KEYS = {
  backendUrl: "sumforu.backendUrl",
  installId: "sumforu.installId",
  persona: "sumforu.persona"
};

export const DEFAULT_BACKEND_URL = "http://127.0.0.1:8788";
export const DEFAULT_MODEL = "gpt-5-mini";

export const DEFAULT_PERSONA =
  "A practical shopper who wants good long-term value, low hassle, and clear tradeoffs before buying.";

export const RECOMMENDATION_LABELS = [
  "Strong fit",
  "Good fit",
  "Mixed",
  "Not recommended"
];

export const MAX_EVIDENCE_CHARS = 7000;
