/**
 * The chapters published on the site, in reading order. This manifest is the
 * single place that maps a source file in book/ to its URL slug and title, and
 * it drives both the sync script and the site's navigation.
 *
 * This module is pure data (no Node APIs) so it can be imported from client
 * components like the sidebar. Filesystem access lives in lib/content.ts.
 */
export const CHAPTERS = [
  { file: "ch1-llm.md", slug: "1-your-first-llm-call", title: "Your First LLM Call" },
  { file: "ch2-conversation.md", slug: "2-conversation", title: "Conversation" },
  { file: "ch3-context.md", slug: "3-context", title: "The System Prompt and Context Builder" },
  { file: "ch4-streaming.md", slug: "4-streaming", title: "Streaming Responses" },
  { file: "ch5-providers.md", slug: "5-provider-abstraction", title: "Provider Abstraction" },
  { file: "ch6-loop.md", slug: "6-the-loop", title: "The Loop" },
  { file: "ch7-tools.md", slug: "7-tool-calling", title: "Tool Calling" },
  { file: "ch8-observation.md", slug: "8-observation-and-continuation", title: "Observation and Continuation" },
  { file: "ch9-stopping.md", slug: "9-stopping-conditions", title: "Stopping Conditions" },
  { file: "ch10-heartbeat.md", slug: "10-heartbeat-and-long-running-tasks", title: "Heartbeat & Long-Running Tasks" },
] as const;

export type Chapter = (typeof CHAPTERS)[number];

export function getChapterBySlug(slug: string): Chapter | undefined {
  return CHAPTERS.find((c) => c.slug === slug);
}

export function getChapterNeighbors(slug: string): {
  prev: Chapter | null;
  next: Chapter | null;
} {
  const i = CHAPTERS.findIndex((c) => c.slug === slug);
  return {
    prev: i > 0 ? CHAPTERS[i - 1] : null,
    next: i >= 0 && i < CHAPTERS.length - 1 ? CHAPTERS[i + 1] : null,
  };
}
