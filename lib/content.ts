import fs from "node:fs";
import path from "node:path";

const CONTENT_DIR = path.join(process.cwd(), "content");

/**
 * Read a chapter's markdown from content/, returning the body with the leading
 * `# Chapter N: …` H1 stripped (the title is rendered separately in the page
 * header so it isn't shown twice). Server-only — uses node:fs.
 */
export function getChapterMarkdown(file: string): string {
  const raw = fs.readFileSync(path.join(CONTENT_DIR, file), "utf8");
  return stripLeadingH1(raw);
}

function stripLeadingH1(markdown: string): string {
  const lines = markdown.split("\n");
  const firstNonEmpty = lines.findIndex((l) => l.trim() !== "");
  if (firstNonEmpty !== -1 && /^#\s+/.test(lines[firstNonEmpty])) {
    lines.splice(0, firstNonEmpty + 1);
  }
  return lines.join("\n").trimStart();
}
