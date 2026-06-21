// Copies the published chapter markdown from ../book into ./content.
// book/ is the single source of truth; re-run `npm run sync` after editing a
// chapter. The committed content/ copies are what Vercel builds from, so the
// published repo is self-sufficient.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const BOOK_DIR = path.join(ROOT, "..", "book");
const CONTENT_DIR = path.join(ROOT, "content");

// Keep in sync with lib/chapters.ts. Plain JS here so the script has no build step.
const FILES = [
  "ch1-llm.md",
  "ch2-conversation.md",
  "ch3-context.md",
  "ch4-streaming.md",
  "ch5-providers.md",
  "ch6-loop.md",
  "ch7-tools.md",
  "ch8-observation.md",
];

if (!fs.existsSync(BOOK_DIR)) {
  console.error(`Source book directory not found: ${BOOK_DIR}`);
  console.error("Run this from the website/ folder with book/ as a sibling.");
  process.exit(1);
}

fs.mkdirSync(CONTENT_DIR, { recursive: true });

let copied = 0;
for (const file of FILES) {
  const src = path.join(BOOK_DIR, file);
  const dest = path.join(CONTENT_DIR, file);
  if (!fs.existsSync(src)) {
    console.error(`Missing chapter source: ${src}`);
    process.exit(1);
  }
  fs.copyFileSync(src, dest);
  copied++;
  console.log(`  synced ${file}`);
}

console.log(`Synced ${copied} chapter(s) into content/.`);
