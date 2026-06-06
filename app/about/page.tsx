import Link from "next/link";
import Image from "next/image";
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
        <Image
          src="/author.jpg"
          alt="Olga Kuzmich"
          width={220}
          height={293}
          priority
          className="float-none mx-auto mb-6 h-auto w-44 rounded-2xl object-cover shadow-sm sm:float-right sm:mb-4 sm:ml-6 sm:mt-0 sm:w-52"
        />

        <p>
          <strong>Olga Kuzmich</strong> is pursuing a combined BS/MS program at
          Johns Hopkins University in Applied Mathematics and Data Science. She
          wrote this book because it is the one she wished existed when she set
          out to understand how AI agents actually work. This book grows out of
          a simple conviction: the engineering behind a capable agent is far
          less magical than it looks, and anyone willing to build it one piece
          at a time can understand the whole thing.
        </p>

        <p>
          Her hands-on work with agents includes contributing to nanobot — a
          lightweight personal AI agent that serves as a reference codebase in
          this book. She currently works at Dwellwell Analytics Inc. as a Data
          Science Engineering Intern, merging AI, hardware, and software to
          transform ordinary houses into data-driven, self-diagnosing
          environments. She writes about her experience as an early-career CS
          professional on her Medium blog,{" "}
          <a
            href="https://medium.com/@okuzmich2005"
            target="_blank"
            rel="noopener noreferrer"
          >
            @okuzmich2005
          </a>
          . Her previous research spans clinical predictive modeling and
          computer vision, with work appearing in venues including{" "}
          <em>Open Forum Infectious Diseases</em> and <em>NeurIPS</em>.
          She is a RISE Global Winner — a Schmidt Futures and Rhodes Trust
          program that selected roughly 100 people from some 80,000 applicants —
          and her early work has been covered by IEEE Entrepreneurship.
        </p>

        <p>She loves oranges, cornflowers, and sparkling water.</p>

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
          . You can also reach Olga by email at{" "}
          <a href="mailto:okuzmic1@jh.edu">okuzmic1@jh.edu</a> or connect on{" "}
          <a
            href="https://www.linkedin.com/in/olgakuzmich/"
            target="_blank"
            rel="noopener noreferrer"
          >
            LinkedIn
          </a>
          .
        </p>
      </div>
    </main>
  );
}
