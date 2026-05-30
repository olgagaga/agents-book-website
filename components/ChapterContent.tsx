"use client";

import { useEffect, useRef } from "react";

/**
 * Renders the build-time-rendered chapter HTML and, after mount, adds a
 * "Copy" button to every code block. Buttons are injected client-side because
 * the markup comes from a static HTML string (dangerouslySetInnerHTML).
 */
export default function ChapterContent({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);

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

  return (
    <div
      ref={ref}
      className="prose prose-neutral max-w-none dark:prose-invert prose-pre:p-0"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
