import { createServer } from "./src/app.js";

const port = Number(process.env.PORT || 8788);
const host = "0.0.0.0";
const app = createServer();

const server = globalThis.Bun
  ? Bun.serve({ port, hostname: host, fetch: app.fetch })
  : null;

if (!server) {
  const { createServer: createNodeServer } = await import("node:http");
  createNodeServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }

    const request = new Request(`http://${req.headers.host}${req.url}`, {
      method: req.method,
      headers: req.headers,
      body: req.method === "GET" || req.method === "HEAD" ? undefined : Buffer.concat(chunks)
    });

    const response = await app.fetch(request);
    res.writeHead(response.status, Object.fromEntries(response.headers));
    res.end(Buffer.from(await response.arrayBuffer()));
  }).listen(port, host, () => {
    console.log(`Summary for You proxy listening on ${host}:${port}`);
  });
}
