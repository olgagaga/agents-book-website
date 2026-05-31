import { createHighlighter, type Highlighter } from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import {
  isAllowedSourcePath,
  langForPath,
  rawUrl,
} from "@/lib/nanobot";

const LANGS = [
  "python",
  "typescript",
  "javascript",
  "markdown",
  "json",
  "toml",
  "yaml",
  "bash",
];
const THEMES = { light: "github-light", dark: "github-dark" } as const;

// A single highlighter is reused across requests (loading Shiki grammars is
// expensive). In dev, stash it on globalThis so hot-reload doesn't re-create it.
const globalForShiki = globalThis as unknown as {
  __shiki?: Promise<Highlighter>;
};

function getHighlighter(): Promise<Highlighter> {
  if (!globalForShiki.__shiki) {
    // The JavaScript regex engine avoids Shiki's oniguruma WASM, which
    // Next's file tracing does not bundle into the serverless function —
    // so this is what makes highlighting work on Vercel, not just locally.
    globalForShiki.__shiki = createHighlighter({
      themes: Object.values(THEMES),
      langs: LANGS,
      engine: createJavaScriptRegexEngine(),
    });
  }
  return globalForShiki.__shiki;
}

// Wrap each line in an `id="L<n>"` anchor so the client can scroll to it.
const lineAnchors = {
  name: "line-anchors",
  line(node: { properties: Record<string, unknown> }, line: number) {
    node.properties.id = `L${line}`;
  },
};

export async function GET(request: Request): Promise<Response> {
  const path = new URL(request.url).searchParams.get("path") ?? "";

  if (!isAllowedSourcePath(path)) {
    return Response.json({ error: "Path not allowed" }, { status: 400 });
  }

  let code: string;
  try {
    const res = await fetch(rawUrl(path));
    if (!res.ok) {
      return Response.json(
        { error: `Could not fetch source (${res.status})` },
        { status: 502 },
      );
    }
    code = await res.text();
  } catch {
    return Response.json({ error: "Could not fetch source" }, { status: 502 });
  }

  const lang = langForPath(path);
  const highlighter = await getHighlighter();
  const html = highlighter.codeToHtml(code, {
    lang: LANGS.includes(lang) ? lang : "text",
    themes: THEMES,
    defaultColor: false,
    transformers: [lineAnchors],
  });

  return Response.json(
    { html, lang },
    {
      headers: {
        // The pinned commit is immutable, so this never changes.
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    },
  );
}
