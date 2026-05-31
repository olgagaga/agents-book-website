"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { CHAPTERS } from "@/lib/chapters";
import ThemeToggle from "./ThemeToggle";

const COLLAPSE_KEY = "outlineCollapsed";

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false); // mobile menu
  const [collapsed, setCollapsed] = useState(false); // desktop hide/show

  // Remember the desktop collapsed preference across navigations.
  useEffect(() => {
    try {
      setCollapsed(localStorage.getItem(COLLAPSE_KEY) === "1");
    } catch {}
  }, []);

  function setCollapsedPersist(value: boolean) {
    setCollapsed(value);
    try {
      localStorage.setItem(COLLAPSE_KEY, value ? "1" : "0");
    } catch {}
  }

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
        className={`${open ? "block" : "hidden"} border-b md:sticky md:top-0 md:block md:h-screen md:flex-shrink-0 md:overflow-y-auto md:border-b-0 md:border-r ${
          collapsed ? "md:w-12" : "md:w-72"
        }`}
        style={{ borderColor: "var(--border)" }}
      >
        {/* Collapsed rail (desktop only): a single button to reopen the panel. */}
        {collapsed && (
          <div className="hidden md:flex md:flex-col md:items-center md:pt-4">
            <button
              onClick={() => setCollapsedPersist(false)}
              aria-label="Show outline"
              title="Show outline"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border text-sm transition-colors hover:bg-black/5 dark:hover:bg-white/10"
              style={{ borderColor: "var(--border)" }}
            >
              »
            </button>
          </div>
        )}

        {/* Full panel: hidden on desktop when collapsed; always shown on mobile. */}
        <div className={collapsed ? "md:hidden" : ""}>
          <div className="hidden items-center justify-between p-5 md:flex">
            <Link href="/" className="text-lg font-semibold leading-tight">
              AI Agent
              <br />
              Engineering
            </Link>
            <div className="flex items-center gap-2">
              <ThemeToggle />
              <button
                onClick={() => setCollapsedPersist(true)}
                aria-label="Hide outline"
                title="Hide outline"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border text-sm transition-colors hover:bg-black/5 dark:hover:bg-white/10"
                style={{ borderColor: "var(--border)" }}
              >
                «
              </button>
            </div>
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
        </div>
      </aside>
    </>
  );
}
