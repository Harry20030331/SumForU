import { createSummaryHandler } from "./summaryProxy.js";

const corsHeaders = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "content-type,x-sumforu-install-id",
  vary: "origin"
};

export function createServer({ env = process.env, fetchImpl = fetch } = {}) {
  const summaryHandler = createSummaryHandler({ env, fetchImpl });

  return {
    async fetch(request) {
      const url = new URL(request.url);

      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders });
      }

      if (request.method === "GET" && url.pathname === "/healthz") {
        return json({ ok: true });
      }

      if (request.method === "POST" && url.pathname === "/api/summary") {
        const response = await summaryHandler(request);
        return withCors(response);
      }

      return json({ error: "Not found" }, 404);
    }
  };
}

export function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      ...corsHeaders
    }
  });
}

function withCors(response) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders)) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}
