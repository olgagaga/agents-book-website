"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { CHAPTERS } from "@/lib/chapters";
import ThemeToggle from "./ThemeToggle";

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Mobile top bar */}
      <div
        className="sticky top-0 z-30 flex items-center justify-between border-b px-4 py-3 md:hidden"
        style={{ borderColor: "var(--border)", background: "var(--bg)" }}
      >
        <Link href="/" className="font-semibold">
          AI Agent Engineering
        </Link>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => setOpen((v) => !v)}
            aria-label="Toggle chapter menu"
            className="rounded-md border px-3 py-1 text-sm"
            style={{ borderColor: "var(--border)" }}
          >
            Chapters
          </button>
        </div>
      </div>

      <aside
        className={`${
          open ? "block" : "hidden"
        } border-b md:sticky md:top-0 md:block md:h-screen md:w-72 md:flex-shrink-0 md:overflow-y-auto md:border-b-0 md:border-r`}
        style={{ borderColor: "var(--border)" }}
      >
        <div className="hidden items-center justify-between p-5 md:flex">
          <Link href="/" className="text-lg font-semibold leading-tight">
            AI Agent
            <br />
            Engineering
          </Link>
          <ThemeToggle />
        </div>

        <nav className="px-3 pb-6 md:px-5">
          <p
            className="px-2 pb-2 pt-2 text-xs font-semibold uppercase tracking-wide"
            style={{ color: "var(--muted)" }}
          >
            Chapters
          </p>
          <ul className="space-y-1">
            {CHAPTERS.map((c, i) => {
              const href = `/chapters/${c.slug}`;
              const active = pathname === href;
              return (
                <li key={c.slug}>
                  <Link
                    href={href}
                    onClick={() => setOpen(false)}
                    className={`block rounded-md px-2 py-1.5 text-sm transition-colors ${
                      active ? "font-semibold" : "hover:bg-black/5 dark:hover:bg-white/10"
                    }`}
                    style={active ? { background: "var(--code-bg)", color: "var(--accent)" } : undefined}
                  >
                    <span style={{ color: "var(--muted)" }}>{i + 1}.</span> {c.title}
                  </Link>
                </li>
              );
            })}
          </ul>

          <div
            className="mt-6 border-t pt-4"
            style={{ borderColor: "var(--border)" }}
          >
            <Link
              href="/about"
              onClick={() => setOpen(false)}
              className={`block rounded-md px-2 py-1.5 text-sm transition-colors ${
                pathname === "/about"
                  ? "font-semibold"
                  : "hover:bg-black/5 dark:hover:bg-white/10"
              }`}
              style={
                pathname === "/about"
                  ? { background: "var(--code-bg)", color: "var(--accent)" }
                  : undefined
              }
            >
              About the author
            </Link>
          </div>
        </nav>
      </aside>
    </>
  );
}
