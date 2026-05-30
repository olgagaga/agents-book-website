import Link from "next/link";
import type { Metadata } from "next";
import ThemeToggle from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: "About the author",
};

export default function About() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 md:py-24">
      <div className="mb-12 flex items-start justify-between gap-4">
        <div>
          <Link
            href="/"
            className="text-sm underline underline-offset-2"
            style={{ color: "var(--muted)" }}
          >
            ← Back home
          </Link>
          <h1 className="mt-3 text-4xl font-bold tracking-tight md:text-5xl">
            About the author
          </h1>
        </div>
        <ThemeToggle />
      </div>

      <div className="prose prose-neutral max-w-none dark:prose-invert">
        <p>
          <strong>Olga Kuzmich</strong> is a computer science student at Johns
          Hopkins University and a builder of AI systems across education,
          healthcare, and synthetic biology. She wrote this book because it is
          the one she wished existed when she set out to understand how AI agents
          actually work — past the demos, down to the loop, the tools, and the
          plain Python that holds it all together.
        </p>

        <p>
          Her hands-on work with agents includes <em>Sorify</em>, a Telegram AI
          voice agent she built and runs, and her role as CTO of QOS Education.
          She founded the <em>“What are you good at, JHU?”</em> podcast, has led
          teams in Hopkins iGEM and the Synthetic Biology Society, and currently
          works at Dwellwell. Her research spans clinical predictive modeling and
          computer vision, with work appearing in venues including{" "}
          <em>Open Forum Infectious Diseases</em> and NeurIPS.
        </p>

        <p>
          She is a RISE Global Winner — a Schmidt Futures and Rhodes Trust
          program that selected roughly 100 people from some 80,000 applicants —
          and her early work has been covered by Forbes and IEEE Entrepreneurship.
          This book grows out of a simple conviction: the engineering behind a
          capable agent is far less magical than it looks, and anyone willing to
          build it one piece at a time can understand the whole thing.
        </p>

        <h2>Get in touch</h2>
        <p>
          Found a typo, have a question, or want to discuss a chapter? Open an
          issue or start a discussion on the{" "}
          <a
            href={process.env.NEXT_PUBLIC_GITHUB_REPO_URL ?? "#"}
            target="_blank"
            rel="noopener noreferrer"
          >
            book’s GitHub repository
          </a>
          .
        </p>
      </div>
    </main>
  );
}
