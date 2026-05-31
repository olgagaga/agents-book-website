/**
 * A rehype plugin tailored to this book's citation convention:
 *
 *   - Inline citations in the body are written as `[N]` (plain text).
 *   - The `## References` section lists each source as a paragraph that begins
 *     with `[N]` and contains the source URL as an autolink.
 *
 * This plugin builds an N → URL map from the References section, then turns
 * every inline `[N]` in the body into a link that opens that source in a new
 * tab. It also forces all external links (including the reference URLs) to open
 * in a new tab.
 *
 * It deliberately skips `<code>`/`<pre>` (so things like `items[1]` in code are
 * left alone) and the References section itself (so the leading `[N]` of each
 * reference entry stays plain text).
 */

interface HastNode {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

const SKIP_TAGS = new Set(["a", "code", "pre"]);

function textContent(node: HastNode): string {
  if (node.type === "text") return node.value ?? "";
  return (node.children ?? []).map(textContent).join("");
}

function findFirstLink(node: HastNode): HastNode | null {
  if (node.type === "element" && node.tagName === "a") return node;
  for (const child of node.children ?? []) {
    const found = findFirstLink(child);
    if (found) return found;
  }
  return null;
}

function isExternalHref(href: unknown): href is string {
  return typeof href === "string" && /^https?:\/\//i.test(href);
}

function openInNewTab(node: HastNode): void {
  node.properties = {
    ...node.properties,
    target: "_blank",
    rel: "noopener noreferrer",
  };
}

/** Recursively force external links to open in a new tab. */
function markExternalLinks(node: HastNode): void {
  if (node.type === "element" && node.tagName === "a") {
    if (isExternalHref(node.properties?.href)) openInNewTab(node);
  }
  for (const child of node.children ?? []) markExternalLinks(child);
}

/** Split a text value on `[N]` markers, linking those present in `refs`. */
function linkifyText(value: string, refs: Map<string, string>): HastNode[] {
  const out: HastNode[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(value)) !== null) {
    const num = match[1];
    const href = refs.get(num);
    if (!href) continue;
    if (match.index > last) {
      out.push({ type: "text", value: value.slice(last, match.index) });
    }
    out.push({
      type: "element",
      tagName: "a",
      properties: {
        href,
        target: "_blank",
        rel: "noopener noreferrer",
        className: ["citation-ref"],
      },
      children: [{ type: "text", value: `[${num}]` }],
    });
    last = match.index + match[0].length;
  }
  if (out.length === 0) return [{ type: "text", value }];
  if (last < value.length) out.push({ type: "text", value: value.slice(last) });
  return out;
}

/** Replace inline `[N]` markers with links, skipping code and existing links. */
function linkifyCitations(node: HastNode, refs: Map<string, string>): void {
  if (!node.children) return;
  const next: HastNode[] = [];
  for (const child of node.children) {
    if (child.type === "text") {
      next.push(...linkifyText(child.value ?? "", refs));
    } else if (
      child.type === "element" &&
      SKIP_TAGS.has(child.tagName ?? "")
    ) {
      next.push(child);
    } else {
      linkifyCitations(child, refs);
      next.push(child);
    }
  }
  node.children = next;
}

export default function rehypeReferences() {
  return (tree: HastNode) => {
    markExternalLinks(tree);

    // Build the N → URL map from paragraphs in the References section.
    const refs = new Map<string, string>();
    const children = tree.children ?? [];
    let inReferences = false;
    for (const node of children) {
      if (node.type === "element" && /^h[1-6]$/.test(node.tagName ?? "")) {
        inReferences = textContent(node).trim().toLowerCase() === "references";
        continue;
      }
      if (inReferences && node.type === "element" && node.tagName === "p") {
        const numMatch = textContent(node).match(/^\s*\[(\d+)\]/);
        if (numMatch) {
          const link = findFirstLink(node);
          const href = link?.properties?.href;
          if (typeof href === "string") refs.set(numMatch[1], href);
        }
      }
    }

    if (refs.size === 0) return;

    // Linkify inline citations everywhere except the References section itself.
    inReferences = false;
    for (const node of children) {
      if (node.type === "element" && /^h[1-6]$/.test(node.tagName ?? "")) {
        inReferences = textContent(node).trim().toLowerCase() === "references";
        continue;
      }
      if (!inReferences) linkifyCitations(node, refs);
    }
  };
}
