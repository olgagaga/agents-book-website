# Chapter 8: Observation and Continuation

Chapter 7 finished with a working tool-call loop, streaming restored, and multi-tool turns supported. However, when a tool returns a lot of text — a file's contents, an HTTP response, or a long shell output — that text is included in the conversation, and the next iteration sends the whole fragment back to the model, collapsing the context window quickly. The second problem with the current implementation is that `agent_step` discards its scratchpad on every turn, so the model loses sight of its prior tool calls across turn boundaries.

By the end of this chapter, tool results pass through a size budget before reaching the model, and the long-term `messages` list keeps the full context of previous tool calls.

## Observations are load-bearing

In the loop, the model's observation of the world is whatever string sits in the `content` field of a `{"role": "tool", ...}` message. Two things follow. First, the string has to clearly describe a tool output, whether it failed or was truncated. The model's plan for the next iteration depends on whether it thinks the previous step succeeded. Second, the string has a budget: every byte of it costs context-window space, both on the request that contains it and on every subsequent request in the same turn.

The Chapter 7 loop already gets the first part right for the failure case (`f"Error: {type(exc).__name__}: {exc}"`). However, it does not address the budgeting constraints yet.

## The size problem

Imagine a `read_file` tool that takes a path and returns the file's contents. A reasonable agent will use it to read a file, decide the file is interesting, and call it again on a related file. A 5,000-line YAML config returns roughly 200 KB of text. The next iteration's request now includes that 200 KB inside the tool-result block, and every subsequent iteration carries it along too.

The pattern compounds in three ways at once. *Vertically*, every additional iteration carries every prior tool result. *Horizontally*, a single iteration can have multiple tool calls and therefore multiple result blocks. *Across turns*, if the scratchpad is persisted, every prior turn's results ride along into the next turn's prompt. Without a size budget anywhere, a session that started under 1,000 tokens crosses 100,000 tokens in a few minutes of conversation and then fails the next request with a context-window error.

## Truncating observations

Addressing these issues calls for a truncation strategy. Start by adding a small helper in `agent/loop.py` for truncating observations:

```python
MAX_OBSERVATION_CHARS = 4000

def _truncate_observation(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 60
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n\n"
        f"[...truncated: {omitted} characters omitted from the middle...]\n\n"
        f"{text[-tail:]}"
    )
```

Notice that the truncation snips the middle and keeps both ends. Tool results often carry useful structure at both ends: a file's imports at the top and its exports at the bottom; a shell command's invocation line at the start and its exit code at the end; an HTTP response's headers at the top and the closing body fragment at the bottom. Throwing away the middle therefore preserves more signal than throwing away the tail. 

The marker text — `[...truncated: 4923 characters omitted from the middle...]` — tells the model that what it is seeing is incomplete and that the missing bytes were in the middle. Without that signal, the model treats the truncated text as the whole truth, which can mean confidently citing a non-existent line or claiming a file ends somewhere it does not.

`MAX_OBSERVATION_CHARS = 4000` is a reasonable starting budget. Real systems make this configurable per-tool: a `now` result of 30 chars and a `read_file` result of 200 KB do not deserve the same budget. Exercise 1 walks through wiring a per-tool budget onto the `Tool` dataclass, and the production reference shows how nanobot organizes the same idea at scale.

Now apply the helper inside the agent loop, before appending the observation to the messages list. Notice the single new line that runs the observation through the helper:

```python
for tc in reply.tool_calls:
    # --- build observation if-else code ---
    observation = _truncate_observation(observation)         # <-- new
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.name,
        "content": observation,
    })
```

## Errors are observations too

The current error handler is fine for the scale at hand:

```python
observation = f"Error: {type(exc).__name__}: {exc}"
```

It says *what* failed (`KeyError`, `TimeoutError`) and *why* (the exception's message). A couple of error-formatting improvements are worth knowing about even if they are not adopted now: distinguish *non-existent tool* errors from *tool raised exception* errors so the model knows whether to retry the same tool with different arguments or pick a different tool, and add a "did you mean…" hint when a tool name is unknown but a close one exists. Exercise 3 sketches both.

## Continuation across turns

The other Chapter 7 fragility was that the agent discards its intermediate scratchpad on every turn, so only the user's prompt and the final reply ever land in the long-term history. The fix here is to change where the scratchpad lives.

To see what gets thrown away, look at the Chapter 7 `agent_step` shape. The first line copies the caller's history into a local `messages` list; the loop appends freely to that copy — user prompt, assistant tool-call messages, synthetic tool-result observations, intermediate assistant text — but only `reply.text` ever leaves the function:

```python
def agent_step(user_message, history, provider, system, on_text_delta=None) -> str:
    messages = list(history)                          # copy of the long-term history
    messages.append({"role": "user", "content": user_message})
    # … loop …
    return reply.text                                 # final text only
```

The local `messages` is the full conversation the model actually saw on that turn. When `agent_step` returns, it is garbage-collected; everything except the final text disappears.

The caller in `chat()` then rebuilds the long-term history from the outside, with only the user prompt going in and the final reply coming out:

```python
reply = agent_step(user_input, messages, provider, system, on_text_delta=…)
messages.append({"role": "user", "content": user_input})
messages.append({"role": "assistant", "content": reply})
```

The scratchpad has to outlive a single call to `agent_step`. The simplest way to make that happen is to stop copying the caller's list and instead mutate it in place — which is a real change of ownership: in Chapter 7, `chat()` was the sole writer of `messages` and `agent_step` worked on a private copy; from here on, `agent_step` is the writer and `chat()` only reads. Also, since `chat()` is no longer responsible for appending anything to `messages`, `agent_step` has to append the final assistant text itself, immediately before returning.

The parameter `history` becomes `messages`, because it is no longer a read-only input to copy from. The `messages = list(history)` copy at the top is gone. And a fresh `messages.append({"role": "assistant", "content": reply.text})` runs on the path that exits the loop. The function still returns the text — handy for the caller that wants to log a transcript — but the mutation is what makes the change durable. (The `[iter]` development print added in Chapter 7's worked example is also gone)

```python
def agent_step(
    user_message: str,
    messages: list[dict], # mutated in place                             
    provider: Provider,
    system: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    # removed copy
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_ITERATIONS):
        reply = provider.call(
            messages, system=system, tools=TOOLS, on_text_delta=on_text_delta,
        )
        # print statement gone

        if not reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.text}) # added history here
            return reply.text

        # --- tool calls handling ---
```

Now drop both appends in `chat()` and let the return value go unused:

```python
def chat(provider: Provider | None = None) -> None:
    """Run an interactive chat loop, dispatching each user turn to the agent loop."""
    # --- chat code ---
    
    # remove these two appends from the chat function
    # messages.append({"role": "user", "content": user_input})
    # messages.append({"role": "assistant", "content": reply})
```

`chat()` no longer touches `messages` at all — its only job is to read a line from the user, hand it to `agent_step`, and let the streaming callback paint the reply onto the terminal in real time.

## A coherent multi-turn dialogue

Run `uv run main.py`:

```
chat — Ctrl-D or empty line to exit

you: what is the current UTC time?

assistant: The current UTC time is **2026-05-13T15:40:22Z** (May 13, 2026, 3:40 PM UTC).

you: how long has it been since I asked?

assistant: About **41 seconds** since your previous question (15:40:22 → 15:41:03 UTC).

you: ^D
```

The second turn does two interesting things at once. It cites the earlier timestamp verbatim — the actual ISO string the `now` tool produced two messages ago, not a paraphrase. And it issues a fresh `now` call to find the current time. Both depend on the scratchpad being intact across the turn boundary: without it, the model would have neither the prior timestamp nor any reason to think `now` is a tool it has used successfully before.

Compare this with the Chapter 7 trace: in that loop, the second turn had no record of the earlier `now` call, so the model either invented a plausible elapsed time or issued a fresh `now` with nothing to compare it against. The continuation change is what closes that gap.

## The cost: context window pressure

Each turn now appends one user message, one assistant message per iteration, and one tool result per tool call. A six-turn conversation with one tool call per turn lands at roughly 24 messages, plus the system prompt — call it 8,000 tokens of tail. Most of that is fine; the input budget on a current Claude or GPT model is comfortably six figures of tokens [1][2]. 

A pair of pressure-relief mechanisms are out of scope for this chapter, each getting its own treatment later in the book. The first on is token-aware truncation: counting tokens instead of characters, applying budgets per-tool, dropping fields the model does not need. In addition to that, consolidation — summarizing older parts of the conversation into a shorter prose digest — is the large-scale fix, and it is what Chapter 16 builds. 

## Production reference

Open [`nanobot/nanobot/agent/runner.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py). Inside `AgentRunner`, two methods are direct production analogues of what this chapter built.

[`_apply_tool_result_budget`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py#L1004) is the production version of the chapter's `_truncate_observation`, and it differs from the toy in three useful ways. First, the truncation strategy is *tail* truncation — it keeps the head and drops the foot: the underlying [`truncate_text`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/utils/helpers.py#L135) helper (in `nanobot/nanobot/utils/helpers.py`) simply does `text[:max_chars] + "\n... (truncated)"`. The tail bias is a deliberate bet that for the kinds of tools nanobot ships (`read_file`, `exec`, `grep`, `web_fetch`), the *head* of the output is more diagnostic than the *foot* — file headers, command invocations, page titles — and the model can always re-issue a more specific call (e.g. `read_file` with a line range) when it needs the missing bytes. The chapter's middle-snip is the right call for a generic loop with no per-tool knowledge; nanobot can be more opinionated because it knows the tools. Second, the budget lives on `AgentRunSpec.max_tool_result_chars` (default 16,000), configurable per agent rather than as a module-level constant — Exercise 1 walks through a similar idea. Third, before truncating, [`_normalize_tool_result`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py#L883) calls [`maybe_persist_tool_result`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/utils/helpers.py#L229), which writes the *full* content to disk under a stable id; if the model later needs the rest, a separate tool can fetch it. The truncation in the prompt is for context-window hygiene; the truth is preserved elsewhere.

[`_microcompact`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py#L979) handles the *across-iterations* problem that pure truncation does not solve. It walks the message list, keeps the most recent 10 results for a fixed set of "compactable" tools (`read_file`, `exec`, `grep`, `glob`, `web_search`, `web_fetch`, `list_dir`), and replaces older entries with a one-line placeholder like `[read_file result omitted from context]`. The rationale matches the chapter's intuition: the model is overwhelmingly reasoning about its most recent observations, so older results from chatty tools can be paged out cheaply. Smaller per-iteration trims, accumulating across a long session.

One production nuance is worth pulling out, since it is what makes the long-running case survivable.

**Auto-compaction on idle.** [`nanobot/nanobot/agent/autocompact.py::AutoCompact`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/autocompact.py#L16) watches sessions for inactivity. When a session has been idle for `session_ttl_minutes` (or when the running context is approaching the model's window), [`Consolidator.maybe_consolidate_by_tokens`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/memory.py#L567) summarizes the older portion of the history with a separate model call, replaces it with a single synthetic message that captures the essentials, and persists the result. The next user turn starts from the summary plus the most recent few exchanges — the agent retains the *gist* of an hour-long conversation without paying the full token cost on every turn. Chapter 16 builds a small version of this.

## Exercises

1. **Per-tool budgets.** Replace `MAX_OBSERVATION_CHARS` with a per-tool budget on the `Tool` dataclass — add a `max_observation_chars: int = 4000` field with a default. Update the loop to read each tool's own budget when truncating. Try it: give `wordcount` a budget of 50 (it returns at most a small integer; 50 is plenty) and `now` a budget of 100. Add a fictional `lorem` tool that returns 50,000 characters of placeholder text and give it a budget of 8,000 to see truncation in action.

2. **Token-aware truncation.** Install `tiktoken` for OpenAI and `anthropic`'s built-in token counter for Claude (the Claude API exposes a token-counting endpoint on the Messages API). Rewrite `_truncate_observation` to take a `max_tokens` argument and use whichever counter matches the active provider. Compare the truncated output to the character-based version on a 5,000-character tool result; how different are the cuts?

3. **Smarter error observations.** Improve the error formatting in two ways: when the model calls a tool that does not exist, suggest the closest valid tool name using `difflib.get_close_matches`. When a tool raises an exception, include the exception class *and* the first line of the traceback (use `traceback.format_exception_only`). Verify both branches by deliberately calling a typo'd tool name and by raising a `RuntimeError` from inside a tool's `run`.

4. **Keep the return value.** `chat()` currently throws away `agent_step`'s return value. Change `chat()` to capture the returned text and write a single line to a `transcript.jsonl` file per user turn, with the user prompt, the final assistant reply, and the iteration count. Use it to spot-check sessions where the iteration count was unusually high (an agent struggling) or unusually low (the agent answering trivially without calling tools).

5. **Stretch: run the budget math.** Pick a real session with the agent and count the token cost of each turn before and after Chapter 8's continuation change. Use the relevant token counter from Exercise 2. Plot the per-turn token count against turn number for a 10-turn session. Where does the curve start bending? Is it linear, super-linear, or sub-linear? (The shape depends on whether tool calls are bursty or evenly distributed.)

6. **Stretch: read [`_apply_tool_result_budget`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py#L1004) and [`_microcompact`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py#L979).** Open [`nanobot/nanobot/agent/runner.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py) and read both methods, along with the [`truncate_text`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/utils/helpers.py#L135) helper in `nanobot/nanobot/utils/helpers.py` and [`maybe_persist_tool_result`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/utils/helpers.py#L229) referenced from [`_normalize_tool_result`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/agent/runner.py#L883). Together they describe a different design from the one this chapter built: tail truncation in place of middle-snip, a per-agent budget, full results persisted to disk so the model can fetch them later, and a separate compaction pass that replaces older entries from a curated set of chatty tools with one-line placeholders. Pick one of those decisions and write a paragraph on whether it is worth adopting in a personal agent. The right call depends on what the tools do.

## References

[1] *Models overview.* Claude API documentation. <https://platform.claude.com/docs/en/about-claude/models/overview>

[2] *Models.* OpenAI documentation. <https://platform.openai.com/docs/models>
