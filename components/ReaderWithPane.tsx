"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSourcePane } from "./SourcePaneProvider";
import SourcePane from "./SourcePane";

const WIDTH_KEY = "sourcePaneWidth";
const MIN_PCT = 25;
const MAX_PCT = 75;

/**
 * Wraps the chapter reading content. When the source pane is open (wide screens
 * only), it lays the content and pane out as an independent-scroll split that
 * fills the viewport, with a draggable divider to resize the two sides.
 */
export default function ReaderWithPane({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isOpen } = useSourcePane();
  const rowRef = useRef<HTMLDivElement>(null);
  const [paneWidth, setPaneWidth] = useState(50); // percent of the row
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    try {
      const saved = Number(localStorage.getItem(WIDTH_KEY));
      if (saved >= MIN_PCT && saved <= MAX_PCT) setPaneWidth(saved);
    } catch {}
  }, []);

  const onPointerMove = useCallback((e: PointerEvent) => {
    const row = rowRef.current;
    if (!row) return;
    const rect = row.getBoundingClientRect();
    const pct = ((rect.right - e.clientX) / rect.width) * 100;
    const clamped = Math.min(MAX_PCT, Math.max(MIN_PCT, pct));
    setPaneWidth(clamped);
  }, []);

  const stopDrag = useCallback(() => {
    setDragging(false);
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", stopDrag);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    setPaneWidth((w) => {
      try {
        localStorage.setItem(WIDTH_KEY, String(Math.round(w)));
      } catch {}
      return w;
    });
  }, [onPointerMove]);

  const startDrag = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      setDragging(true);
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", stopDrag);
    },
    [onPointerMove, stopDrag],
  );

  useEffect(() => {
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", stopDrag);
    };
  }, [onPointerMove, stopDrag]);

  return (
    <main className="min-w-0 flex-1">
      <div
        ref={rowRef}
        className={isOpen ? "md:sticky md:top-0 md:flex md:h-screen" : ""}
      >
        <div
          className={isOpen ? "min-w-0 md:overflow-y-auto" : ""}
          style={isOpen ? { width: `${100 - paneWidth}%` } : undefined}
        >
          {children}
        </div>

        {isOpen && (
          <>
            {/* Drag handle */}
            <div
              onPointerDown={startDrag}
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize code panel"
              data-dragging={dragging ? "true" : undefined}
              className="source-resize-handle hidden md:block md:flex-shrink-0 md:cursor-col-resize"
            />
            <SourcePane
              className="hidden md:flex md:h-screen md:flex-shrink-0"
              style={{ width: `${paneWidth}%` }}
            />
          </>
        )}
      </div>
    </main>
  );
}
