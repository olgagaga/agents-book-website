"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { parseNanobotBlobUrl, type ParsedSource } from "@/lib/nanobot";

type PaneStatus = "idle" | "loading" | "loaded" | "error";

interface PaneState {
  status: PaneStatus;
  source: ParsedSource | null;
  html: string | null;
  error: string | null;
}

interface SourcePaneContextValue extends PaneState {
  isOpen: boolean;
  /** Open the pane for a nanobot blob URL. Returns false if not a nanobot link. */
  openSource: (href: string) => boolean;
  close: () => void;
}

const SourcePaneContext = createContext<SourcePaneContextValue | null>(null);

export function useSourcePane(): SourcePaneContextValue {
  const ctx = useContext(SourcePaneContext);
  if (!ctx) throw new Error("useSourcePane must be used within SourcePaneProvider");
  return ctx;
}

const CLOSED: PaneState = {
  status: "idle",
  source: null,
  html: null,
  error: null,
};

export default function SourcePaneProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [state, setState] = useState<PaneState>(CLOSED);

  const close = useCallback(() => setState(CLOSED), []);

  const openSource = useCallback((href: string): boolean => {
    const source = parseNanobotBlobUrl(href);
    if (!source) return false;

    setState({ status: "loading", source, html: null, error: null });

    fetch(`/api/source?path=${encodeURIComponent(source.path)}`)
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error ?? `Request failed (${res.status})`);
        return data.html as string;
      })
      .then((html) => {
        // Ignore if the user already navigated to a different source/closed.
        setState((prev) =>
          prev.source?.githubUrl === source.githubUrl
            ? { status: "loaded", source, html, error: null }
            : prev,
        );
      })
      .catch((err: Error) => {
        setState((prev) =>
          prev.source?.githubUrl === source.githubUrl
            ? { status: "error", source, html: null, error: err.message }
            : prev,
        );
      });

    return true;
  }, []);

  const value = useMemo<SourcePaneContextValue>(
    () => ({ ...state, isOpen: state.source !== null, openSource, close }),
    [state, openSource, close],
  );

  return (
    <SourcePaneContext.Provider value={value}>
      {children}
    </SourcePaneContext.Provider>
  );
}
