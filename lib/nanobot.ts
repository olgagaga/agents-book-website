/**
 * The pinned nanobot reference codebase. The book cites this exact commit, and
 * the source pane renders files from it. Keeping the repo/SHA here means the
 * pin lives in one place on the website side.
 */
export const NANOBOT_REPO = "HKUDS/nanobot";
export const NANOBOT_SHA = "28f9bbff314cf90b0401b3aa220ca7a723c4f4ab";

export const NANOBOT_BLOB_PREFIX = `https://github.com/${NANOBOT_REPO}/blob/${NANOBOT_SHA}/`;

export interface ParsedSource {
  /** Repo-relative path, e.g. "nanobot/providers/anthropic_provider.py". */
  path: string;
  /** Just the file name, e.g. "anthropic_provider.py". */
  fileName: string;
  /** 1-based first highlighted line, if the link had an #L anchor. */
  fromLine?: number;
  /** 1-based last highlighted line (equals fromLine for a single line). */
  toLine?: number;
  /** The original GitHub URL, for the "Open on GitHub" button. */
  githubUrl: string;
}

const BLOB_RE = new RegExp(
  `^https://github\\.com/${NANOBOT_REPO}/blob/${NANOBOT_SHA}/` +
    `([^#?]+?)(?:#L(\\d+)(?:-L(\\d+))?)?$`,
);

/** Parse a pinned nanobot blob URL into its path and line range, or null. */
export function parseNanobotBlobUrl(href: string): ParsedSource | null {
  const m = BLOB_RE.exec(href);
  if (!m) return null;
  const path = m[1];
  const fromLine = m[2] ? parseInt(m[2], 10) : undefined;
  const toLine = m[3] ? parseInt(m[3], 10) : fromLine;
  return {
    path,
    fileName: path.split("/").pop() ?? path,
    fromLine,
    toLine,
    githubUrl: href,
  };
}

/** True if a path is safe to fetch: inside the nanobot package, no traversal. */
export function isAllowedSourcePath(path: string): boolean {
  return /^nanobot\/[A-Za-z0-9._/-]+\.[A-Za-z0-9]+$/.test(path) && !path.includes("..");
}

/** The raw.githubusercontent URL for a repo-relative path at the pinned commit. */
export function rawUrl(path: string): string {
  return `https://raw.githubusercontent.com/${NANOBOT_REPO}/${NANOBOT_SHA}/${path}`;
}

/** Map a file extension to a Shiki language id. */
export function langForPath(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "py":
      return "python";
    case "ts":
    case "tsx":
      return "typescript";
    case "js":
    case "mjs":
      return "javascript";
    case "md":
      return "markdown";
    case "json":
      return "json";
    case "toml":
      return "toml";
    case "yml":
    case "yaml":
      return "yaml";
    case "sh":
      return "bash";
    default:
      return "text";
  }
}
