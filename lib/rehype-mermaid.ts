import { visit } from "unist-util-visit";
import type { Element, Root, Text } from "hast";

/**
 * Rehype plugin that rewrites fenced ```mermaid blocks into a plain
 * `<div class="mermaid-diagram">` carrying the raw diagram source as text.
 *
 * It must run BEFORE rehype-pretty-code so Shiki never sees the `mermaid`
 * language (which it cannot highlight) and never wraps the source in token
 * spans. The diagram is rendered to SVG on the client by ChapterContent,
 * which reads the element's textContent and feeds it to mermaid.
 */
export default function rehypeMermaid() {
  return (tree: Root) => {
    visit(tree, "element", (node: Element) => {
      if (node.tagName !== "pre") return;

      const code = node.children.find(
        (child): child is Element =>
          child.type === "element" && child.tagName === "code",
      );
      if (!code) return;

      const className = code.properties?.className;
      const classes = Array.isArray(className) ? className.map(String) : [];
      if (!classes.includes("language-mermaid")) return;

      const source = collectText(code);

      // Rewrite the <pre> in place into a non-<pre> container so neither
      // rehype-pretty-code nor the copy-button pass (which targets <pre>)
      // touches it.
      node.tagName = "div";
      node.properties = { className: ["mermaid-diagram"] };
      node.children = [{ type: "text", value: source } as Text];
    });
  };
}

function collectText(node: Element | Text): string {
  if (node.type === "text") return node.value;
  if (!("children" in node)) return "";
  return node.children
    .map((child) =>
      child.type === "text"
        ? child.value
        : child.type === "element"
          ? collectText(child)
          : "",
    )
    .join("");
}
