"use client";

import { useEffect, useRef } from "react";
import { useSourcePane } from "./SourcePaneProvider";

export default function SourcePane({
  className = "",
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  const { status, source, html, error, close } = useSourcePane();
  const bodyRef = useRef<HTMLDivElement>(null);

  // After the code renders, highlight the cited line range and scroll to it.
  useEffect(() => {
    if (status !== "loaded" || !source?.fromLine || !bodyRef.current) return;
    const root = bodyRef.current;
    const from = source.fromLine;
    const to = source.toLine ?? from;
    for (let n = from; n <= to; n++) {
      root.querySelector(`#L${n}`)?.classList.add("source-line-highlight");
    }
    const target = root.querySelector<HTMLElement>(`#L${from}`);
    if (target) {
      // Center the cited line in the pane.
      target.scrollIntoView({ block: "center" });
    }
  }, [status, source, html]);

  if (!source) return null;

  const lineLabel = source.fromLine
    ? source.toLine && source.toLine !== source.fromLine
      ? `lines ${source.fromLine}–${source.toLine}`
      : `line ${source.fromLine}`
    : null;

  return (
    <aside
      className={`flex flex-col border-l ${className}`}
      style={{ borderColor: "var(--border)", background: "var(--bg)", ...style }}
    >
      <header
        className="flex items-center justify-between gap-3 border-b px-4 py-3"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="min-w-0">
          <p className="truncate font-mono text-sm font-medium">{source.fileName}</p>
          {lineLabel && (
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              {lineLabel}
            </p>
          )}
        </div>
        <div className="flex flex-shrink-0 items-center gap-2">
          <a
            href={source.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md border px-2.5 py-1 text-xs font-medium transition-colors hover:bg-black/5 dark:hover:bg-white/10"
            style={{ borderColor: "var(--border)", color: "var(--accent)" }}
          >
            GitHub ↗
          </a>
          <button
            onClick={close}
            aria-label="Close code panel"
            className="inline-flex h-7 w-7 items-center justify-center rounded-md border text-sm transition-colors hover:bg-black/5 dark:hover:bg-white/10"
            style={{ borderColor: "var(--border)" }}
          >
            ✕
          </button>
        </div>
      </header>

      <div ref={bodyRef} className="source-pane min-h-0 flex-1 overflow-auto">
        {status === "loading" && (
          <p className="p-4 text-sm" style={{ color: "var(--muted)" }}>
            Loading {source.fileName}…
          </p>
        )}
        {status === "error" && (
          <p className="p-4 text-sm" style={{ color: "var(--muted)" }}>
            Couldn’t load the source ({error}).{" "}
            <a
              href={source.githubUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
              style={{ color: "var(--accent)" }}
            >
              Open on GitHub ↗
            </a>
          </p>
        )}
        {status === "loaded" && html && (
          <div dangerouslySetInnerHTML={{ __html: html }} />
        )}
      </div>
    </aside>
  );
}
