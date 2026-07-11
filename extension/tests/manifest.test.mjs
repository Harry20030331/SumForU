import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { stat } from "node:fs/promises";

const manifest = JSON.parse(await readFile(new URL("../manifest.json", import.meta.url), "utf8"));

test("manifest is ready for Chrome Web Store review", () => {
  assert.equal(manifest.name, "SumForU");
  assert.deepEqual(manifest.permissions.sort(), ["activeTab", "scripting", "storage"]);
  assert.equal(manifest.host_permissions, undefined);
  assert.equal(manifest.content_scripts, undefined);
  assert.deepEqual(Object.keys(manifest.icons).sort(), ["128", "16", "32", "48"]);
  assert.equal(manifest.icons["128"], "assets/icons/icon-128.png");
});

test("manifest icon files exist", async () => {
  for (const path of Object.values(manifest.icons)) {
    const info = await stat(new URL(`../${path}`, import.meta.url));
    assert.equal(info.isFile(), true);
    assert.equal(info.size > 0, true);
  }
});
