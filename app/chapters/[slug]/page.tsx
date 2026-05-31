import { notFound } from "next/navigation";
import type { Metadata } from "next";
import Sidebar from "@/components/Sidebar";
import PrevNext from "@/components/PrevNext";
import Comments from "@/components/Comments";
import ChapterContent from "@/components/ChapterContent";
import SourcePaneProvider from "@/components/SourcePaneProvider";
import ReaderWithPane from "@/components/ReaderWithPane";
import {
  CHAPTERS,
  getChapterBySlug,
  getChapterNeighbors,
} from "@/lib/chapters";
import { getChapterMarkdown } from "@/lib/content";
import { renderMarkdown } from "@/lib/markdown";

export function generateStaticParams() {
  return CHAPTERS.map((c) => ({ slug: c.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const chapter = getChapterBySlug(slug);
  return { title: chapter?.title ?? "Chapter" };
}

export default async function ChapterPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const chapter = getChapterBySlug(slug);
  if (!chapter) notFound();

  const number = CHAPTERS.findIndex((c) => c.slug === slug) + 1;
  const html = await renderMarkdown(getChapterMarkdown(chapter.file));
  const { prev, next } = getChapterNeighbors(slug);

  return (
    <SourcePaneProvider>
      <div className="md:flex">
        <Sidebar />
        <ReaderWithPane>
          <article className="mx-auto max-w-3xl px-5 py-10 md:px-8 md:py-16">
            <header className="mb-10">
              <p
                className="text-sm font-semibold uppercase tracking-wide"
                style={{ color: "var(--muted)" }}
              >
                Chapter {number}
              </p>
              <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">
                {chapter.title}
              </h1>
            </header>

            <ChapterContent html={html} />

            <PrevNext prev={prev} next={next} />
            <Comments chapterTitle={`Chapter ${number}: ${chapter.title}`} />
          </article>
        </ReaderWithPane>
      </div>
    </SourcePaneProvider>
  );
}
