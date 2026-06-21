"use client";

import { useEffect, useRef } from "react";
import { NANOBOT_BLOB_PREFIX } from "@/lib/nanobot";
import { useSourcePane } from "./SourcePaneProvider";

/**
 * Renders the build-time-rendered chapter HTML and, after mount, adds a
 * "Copy" button to every code block. Buttons are injected client-side because
 * the markup comes from a static HTML string (dangerouslySetInnerHTML).
 *
 * It also intercepts clicks on nanobot source links so they open the in-app
 * source pane (on wide screens) instead of navigating away.
 */
export default function ChapterContent({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const { openSource } = useSourcePane();

  // Intercept nanobot blob-link clicks → open the source pane.
  useEffect(() => {
    const root = ref.current;
    if (!root) return;

    function onClick(e: MouseEvent) {
      // Let modified clicks (new tab/window) and non-primary buttons through.
      if (e.defaultPrevented || e.button !== 0) return;
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      // No room to split on narrow screens — let it open GitHub in a new tab.
      if (!window.matchMedia("(min-width: 768px)").matches) return;

      const link = (e.target as HTMLElement).closest("a");
      const href = link?.getAttribute("href");
      if (!href || !href.startsWith(NANOBOT_BLOB_PREFIX)) return;

      if (openSource(href)) e.preventDefault();
    }

    root.addEventListener("click", onClick);
    return () => root.removeEventListener("click", onClick);
  }, [openSource]);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;

    const blocks = root.querySelectorAll<HTMLPreElement>("pre");
    blocks.forEach((pre) => {
      if (pre.dataset.copyReady) return;
      pre.dataset.copyReady = "true";
      pre.style.position = "relative";

      const button = document.createElement("button");
      button.type = "button";
      button.className = "code-copy-btn";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy code to clipboard");

      button.addEventListener("click", async () => {
        const code = pre.querySelector("code");
        const text = (code ?? pre).textContent ?? "";
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = "Copied!";
          button.classList.add("is-copied");
          window.setTimeout(() => {
            button.textContent = "Copy";
            button.classList.remove("is-copied");
          }, 1500);
        } catch {
          button.textContent = "Failed";
          window.setTimeout(() => {
            button.textContent = "Copy";
          }, 1500);
        }
      });

      pre.appendChild(button);
    });
  }, [html]);

  // Render ```mermaid blocks (rewritten to `.mermaid-diagram` divs at build
  // time) into SVG on the client, and re-render them when the theme changes
  // so colors track light/dark mode.
  useEffect(() => {
    const root = ref.current;
    if (!root) return;

    let cancelled = false;

    async function render() {
      const nodes = Array.from(
        root!.querySelectorAll<HTMLElement>(".mermaid-diagram"),
      );
      if (!nodes.length) return;

      const mermaid = (await import("mermaid")).default;
      if (cancelled) return;

      const dark = document.documentElement.classList.contains("dark");
      mermaid.initialize({
        startOnLoad: false,
        theme: dark ? "dark" : "default",
        securityLevel: "loose", // book-authored content; allows <br/> in labels
        flowchart: { htmlLabels: true },
        themeVariables: { fontSize: "16px" },
      });

      for (const node of nodes) {
        // Stash the original source once; mermaid replaces the content with
        // SVG, so a re-render (e.g. on theme toggle) needs the source back.
        if (node.dataset.src === undefined) {
          node.dataset.src = node.textContent ?? "";
        }
        node.removeAttribute("data-processed");
        node.textContent = node.dataset.src;
      }

      try {
        await mermaid.run({ nodes });
      } catch {
        // Leave the raw source visible if a diagram fails to parse.
      }
    }

    render();

    // Re-render when the `.dark` class on <html> flips.
    const observer = new MutationObserver(() => render());
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, [html]);

  return (
    <div
      ref={ref}
      className="prose prose-neutral max-w-none dark:prose-invert prose-pre:p-0"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
