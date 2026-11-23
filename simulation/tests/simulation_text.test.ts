import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import {
  deepSanitizeSim,
  sanitizeSimText,
} from "../../src/pages/simulationText.ts";

const SIM_DIR = path.resolve(import.meta.dirname, "../../public/sim");

function collectStrings(o: unknown, out: string[] = []): string[] {
  if (typeof o === "string") out.push(o);
  else if (Array.isArray(o)) o.forEach((v) => collectStrings(v, out));
  else if (o && typeof o === "object")
    Object.values(o).forEach((v) => collectStrings(v, out));
  return out;
}

test("sanitiser converts known LaTeX forms to plain text", () => {
  assert.equal(sanitizeSimText("$S_{\\max}$"), "S_max");
  assert.equal(sanitizeSimText("$\\tan(\\beta)$"), "tan(beta)");
  assert.equal(
    sanitizeSimText("$\\sigma = H / (\\tan(\\beta)\\sqrt{2\\pi})$"),
    "sigma = H / (tan(beta)*sqrt(2*pi))"
  );
});

test("no rendered simulation string contains '$' or a backslash", () => {
  const files = fs.readdirSync(SIM_DIR).filter((f) => f.endsWith(".json"));
  assert.ok(files.length > 0);
  for (const f of files) {
    const json = JSON.parse(fs.readFileSync(path.join(SIM_DIR, f), "utf8"));
    for (const s of collectStrings(deepSanitizeSim(json))) {
      assert.ok(!s.includes("$"), `${f}: contains $: ${s}`);
      assert.ok(!s.includes("\\"), `${f}: contains backslash: ${s}`);
    }
  }
});
