"use client";

import { useEffect, useRef } from "react";

const REPO = process.env.NEXT_PUBLIC_GISCUS_REPO;
const REPO_ID = process.env.NEXT_PUBLIC_GISCUS_REPO_ID;
const CATEGORY = process.env.NEXT_PUBLIC_GISCUS_CATEGORY ?? "General";
const CATEGORY_ID = process.env.NEXT_PUBLIC_GISCUS_CATEGORY_ID;
const REPO_URL = process.env.NEXT_PUBLIC_GITHUB_REPO_URL;

function issueUrl(chapterTitle: string): string | null {
  if (!REPO_URL) return null;
  const params = new URLSearchParams({
    title: `Feedback: ${chapterTitle}`,
    labels: "feedback",
  });
  return `${REPO_URL}/issues/new?${params.toString()}`;
}

export default function Comments({ chapterTitle }: { chapterTitle: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const configured = Boolean(REPO && REPO_ID && CATEGORY_ID);

  useEffect(() => {
    if (!configured || !ref.current || ref.current.childElementCount > 0) return;

    const theme = document.documentElement.classList.contains("dark")
      ? "dark"
      : "light";

    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    script.async = true;
    script.crossOrigin = "anonymous";
    script.setAttribute("data-repo", REPO!);
    script.setAttribute("data-repo-id", REPO_ID!);
    script.setAttribute("data-category", CATEGORY);
    script.setAttribute("data-category-id", CATEGORY_ID!);
    script.setAttribute("data-mapping", "pathname");
    script.setAttribute("data-strict", "0");
    script.setAttribute("data-reactions-enabled", "1");
    script.setAttribute("data-emit-metadata", "0");
    script.setAttribute("data-input-position", "top");
    script.setAttribute("data-theme", theme);
    script.setAttribute("data-lang", "en");
    ref.current.appendChild(script);
  }, [configured]);

  const url = issueUrl(chapterTitle);

  return (
    <section
      className="mt-16 border-t pt-8"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-xl font-semibold">Discussion</h2>
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm underline underline-offset-2"
            style={{ color: "var(--accent)" }}
          >
            Found a typo? Open an issue ↗
          </a>
        )}
      </div>

      {configured ? (
        <div ref={ref} />
      ) : (
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Comments are not configured yet. Set the <code>NEXT_PUBLIC_GISCUS_*</code>{" "}
          environment variables (see <code>.env.example</code>) to enable
          GitHub-backed discussion here.
        </p>
      )}
    </section>
  );
}
