# Sum for You Backend Proxy

This Render-ready Node service keeps the OpenAI API key on the server and exposes the extension-facing endpoint.

## Endpoints

- `GET /healthz` returns `{ "ok": true }` for Render health checks.
- `POST /api/summary` accepts `{ persona, context }` and returns `{ result }`.

## Environment

- `OPENAI_API_KEY`: required, set as a Render secret.
- `OPENAI_MODEL`: optional, defaults to `gpt-5-mini`.
- `SUMFORU_DAILY_LIMIT`: optional, defaults to `30` summaries per anonymous extension install id per day per running instance.
- `PORT`: provided by Render.

## Local Run

```bash
OPENAI_API_KEY=sk-your-key npm start
```

## Render Deployment

The root `render.yaml` defines a single free Node web service named `sumforu-api`.

1. Push this repo to GitHub.
2. In Render, create a new Blueprint from the repo.
3. Fill `OPENAI_API_KEY` when Render prompts for the `sync: false` secret.
4. After deploy, copy the service URL.
5. Replace the default backend URL in:
   - `extension/src/shared/defaults.js`
   - `extension/src/options/options.js`
6. Reload or package the Chrome extension.
