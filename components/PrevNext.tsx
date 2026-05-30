import Link from "next/link";
import type { Chapter } from "@/lib/chapters";

export default function PrevNext({
  prev,
  next,
}: {
  prev: Chapter | null;
  next: Chapter | null;
}) {
  return (
    <nav
      className="mt-16 flex items-stretch justify-between gap-4 border-t pt-8"
      style={{ borderColor: "var(--border)" }}
    >
      {prev ? (
        <Link
          href={`/chapters/${prev.slug}`}
          className="group flex-1 rounded-lg border p-4 transition-colors hover:bg-black/5 dark:hover:bg-white/10"
          style={{ borderColor: "var(--border)" }}
        >
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            ← Previous
          </span>
          <span className="mt-1 block font-medium">{prev.title}</span>
        </Link>
      ) : (
        <span className="flex-1" />
      )}
      {next ? (
        <Link
          href={`/chapters/${next.slug}`}
          className="group flex-1 rounded-lg border p-4 text-right transition-colors hover:bg-black/5 dark:hover:bg-white/10"
          style={{ borderColor: "var(--border)" }}
        >
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            Next →
          </span>
          <span className="mt-1 block font-medium">{next.title}</span>
        </Link>
      ) : (
        <span className="flex-1" />
      )}
    </nav>
  );
}
