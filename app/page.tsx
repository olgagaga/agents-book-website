import Link from "next/link";
import { CHAPTERS } from "@/lib/chapters";
import ThemeToggle from "@/components/ThemeToggle";

export default function Home() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 md:py-24">
      <div className="mb-12 flex items-start justify-between gap-4">
        <div>
          <p
            className="text-sm font-semibold uppercase tracking-wide"
            style={{ color: "var(--muted)" }}
          >
            A hands-on book
          </p>
          <h1 className="mt-2 text-4xl font-bold tracking-tight md:text-5xl">
            AI Agent Engineering
          </h1>
        </div>
        <ThemeToggle />
      </div>

      <p className="text-lg leading-relaxed" style={{ color: "var(--muted)" }}>
        Build a personal AI agent from scratch in Python. Each chapter adds a single idea until you have
        a real agent you understand top to bottom. All file links in Production references are clickable and take you to the exact line in the exact version of the reference codebase. 
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
        <Link
          href={`/chapters/${CHAPTERS[0].slug}`}
          className="inline-flex items-center rounded-lg px-5 py-2.5 font-medium text-white transition-opacity hover:opacity-90"
          style={{ background: "var(--accent)" }}
        >
          Start reading → Chapter 1
        </Link>
        <Link
          href="/about"
          className="text-sm font-medium underline underline-offset-2"
          style={{ color: "var(--accent)" }}
        >
          About the author
        </Link>
      </div>

      <section className="mt-16">
        <h2
          className="mb-4 text-xs font-semibold uppercase tracking-wide"
          style={{ color: "var(--muted)" }}
        >
          Chapters
        </h2>
        <ol className="space-y-1">
          {CHAPTERS.map((c, i) => (
            <li key={c.slug}>
              <Link
                href={`/chapters/${c.slug}`}
                className="flex items-baseline gap-3 rounded-md px-3 py-2.5 transition-colors hover:bg-black/5 dark:hover:bg-white/10"
              >
                <span
                  className="text-sm font-semibold tabular-nums"
                  style={{ color: "var(--muted)" }}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="font-medium">{c.title}</span>
              </Link>
            </li>
          ))}
        </ol>
        <p className="mt-8 text-sm" style={{ color: "var(--muted)" }}>
          The first eight chapters are published here. More are on the way.
        </p>
      </section>

      <section
        className="mt-16 rounded-xl border p-5 md:p-6"
        style={{ borderColor: "var(--border)", background: "var(--code-bg)" }}
      >
        <h2 className="text-base font-semibold">About the reference codebase</h2>
        <p className="mt-3 text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
          Each chapter ends with a <em>Production reference</em> that maps the toy
          code you build to real functions in{" "}
          <a
            href="https://github.com/HKUDS/nanobot"
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2"
            style={{ color: "var(--accent)" }}
          >
            nanobot
          </a>
          , an open-source personal AI agent. 
          nanobot is under active development, so if a section the book references
          has since changed upstream, you can always read the exact version the
          book is describing at the pinned commit:
        </p>
        <p className="mt-3 text-sm">
          <a
            href="https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab"
            target="_blank"
            rel="noopener noreferrer"
            className="break-all underline underline-offset-2"
            style={{ color: "var(--accent)" }}
          >
            github.com/HKUDS/nanobot/tree/28f9bbf
          </a>
        </p>
        <p className="mt-3 text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
          To check it out locally:
        </p>
        <pre
          className="mt-2 overflow-x-auto rounded-lg border p-3 text-xs leading-relaxed"
          style={{ borderColor: "var(--border)", background: "var(--bg)" }}
        >
          <code>{`git clone https://github.com/HKUDS/nanobot.git
cd nanobot
git checkout 28f9bbf`}</code>
        </pre>
      </section>
    </main>
  );
}
