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
        a real agent you understand top to bottom.
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
          The first five chapters are published here. More are on the way.
        </p>
      </section>
    </main>
  );
}
