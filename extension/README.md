# Sum for You Chrome Extension Demo

This folder contains a Chrome extension demo for persona-aware product summaries. It is separate from the research pipeline and calls the Sum for You backend proxy, which keeps the OpenAI API key on the server.

Public listing: [SumForU on the Chrome Web Store](https://chromewebstore.google.com/detail/sumforu/giilohlnioiocngbmbdjknpegjckoeeh?utm_source=ext_app_menu)

## Load The Extension

1. Open `chrome://extensions`.
2. Turn on Developer mode.
3. Choose Load unpacked.
4. Select `/Users/yumingfeng/repo/SumForU/extension`.

## Configure Persona

Open the extension settings page and write the default shopper persona once. Future summaries use this persona unless the user edits it in the popup or settings.

For local development, the backend URL defaults to `http://127.0.0.1:8788`. For a Chrome Web Store build, set the default backend URL in `extension/src/shared/defaults.js` and `extension/src/options/options.js` to the deployed Render service URL.

## Run The Backend Locally

The extension calls `POST /api/summary` on the backend. To run it locally:

```bash
OPENAI_API_KEY=sk-your-key npm start
```

The backend uses `gpt-5-mini` by default and limits each anonymous extension install id to 30 summaries per day per running instance.

## Try Demo Pages

Use any product or hotel page, including this live Hotels.com test case:

- `https://www.hotels.com/ho557438/innside-by-melia-new-york-nomad-new-york-united-states-of-america/`

Or start with the reproducible local fixtures:

- `extension/demo-pages/electronics-headphones.html`
- `extension/demo-pages/kitchen-blender.html`
- `extension/demo-pages/ambiguous-smart-lamp.html`

For `file://` pages, Chrome may require enabling Allow access to file URLs for the unpacked extension. A simple alternative is to serve only the demo pages locally:

```bash
python3 -m http.server 8787 --bind 127.0.0.1 --directory extension/demo-pages
```

Then open:

- `http://127.0.0.1:8787/electronics-headphones.html`
- `http://127.0.0.1:8787/kitchen-blender.html`
- `http://127.0.0.1:8787/ambiguous-smart-lamp.html`

## Expected Flow

1. Open a product page.
2. Click the Sum for You extension.
3. Edit the shopper persona if needed.
4. Click Generate.
5. Review the product summary, persona fit, strengths, concerns, score, recommendation, and grounding note.

## Chrome Web Store Caveat

The public release should point to the deployed Render backend so users never handle an OpenAI API key. A public release may need a privacy policy because product page content can be sent to the backend and OpenAI for summarization.
