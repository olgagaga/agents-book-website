# Chapter 4: Streaming Responses

The `chat()` function in `main.py` waits for the model's full reply before printing anything. That is not how the model actually works — it generates tokens one at a time, and the first word is ready within a few hundred milliseconds. Our code is sitting on it.

By the end of this chapter, `llm()` will be a Python generator that yields text deltas as they arrive, `chat()` will render them live, and the system prompt will be marked for prompt caching so we are not paying full price to re-process it on every turn.

## Why streaming matters

Three things change when you turn streaming on:

1. **Time to first token drops dramatically.** Without streaming, the user waits for the entire response. With streaming, they see the first words within a few hundred milliseconds. The total time to finish the reply is the same, but the *perceived* latency — how long the agent feels frozen — collapses to nearly zero.

2. **You can act on partial output.** The CLI tool will only use streaming for display, but agents that route output to other places — a Telegram channel that shows typing indicators, a UI that renders Markdown progressively, a downstream tool that begins work on the first paragraph — all benefit. We will exploit this much more aggressively in Chapter 22.

3. **Tool calls stream too — and that matters more than it sounds.** Once we add tools (Chapter 7), the model emits the tool name first, then the arguments piece by piece. Without streaming, a tool whose arguments are a 2-KB shell command freezes the agent for seconds while you wait for the closing brace. With streaming, you can render `calling read_file('notes.txt')...` the moment the function name commits, and — critically — the user has a window to hit Ctrl-C if they see the agent about to do something destructive. Chapter 9 builds the interrupt path properly.

## How streaming works, mechanically

Streaming is built on **server-sent events** (SSE) [1], a thin protocol on top of HTTP. The client opens a single long-lived HTTP connection. The server keeps the connection open and writes events to it as they happen. Each event has a name and a JSON data payload, separated by blank lines:

```
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}
```

You can read raw SSE with nothing more than a streaming HTTP client. We will not — the SDK does this for us — but it is good to know that it is plain text over an HTTP connection.

Anthropic's stream emits a fixed sequence of event types per response [2]:

| Event | When it fires | What's in it |
|---|---|---|
| `message_start` | Once, at the beginning | Empty message metadata |
| `content_block_start` | Each time a new content block begins | The block's type (`text`, `tool_use`, `thinking`) |
| `content_block_delta` | Many times per block | A small piece of the block's content |
| `content_block_stop` | Each time a block ends | Nothing — just a marker |
| `message_delta` | Near the end | Final `stop_reason`, output token count |
| `message_stop` | Once, at the end | Nothing — just a marker |

For a plain-text reply, the sequence is `message_start → content_block_start (text) → many content_block_deltas → content_block_stop → message_delta → message_stop`. For a reply that includes a tool call, you get a second `content_block_start/delta/stop` cycle for the `tool_use` block. For adaptive thinking, you get a `thinking` block first, then a `text` block.

We are not going to consume raw events directly in this chapter. We will use the SDK's high-level helper, `client.messages.stream(...)`. Under the hood it does three small but tedious things: it opens the SSE connection and parses the `event:` / `data:` lines back into Python objects, it routes each event by type into a typed object so you do not have to pattern-match on raw JSON, and it accumulates deltas as they arrive so a final `get_final_message()` call can hand you a fully assembled `Message` with `content`, `usage`, and `stop_reason` populated. Doing this by hand is not hard — perhaps fifty lines of Python — but it is fifty lines that tend to drift away from the spec as new event types are added (extended thinking, server-side tool calls, refusals). Stretch Exercise 7 has you replace `text_stream` with raw event iteration so you can see the underlying machinery; Stretch Exercise 8 has you parse the SSE protocol from a raw HTTP stream without the SDK at all.

## Refactoring `llm` into a generator

The simplest possible streaming call looks like this:

```python
with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=16000,
    messages=[{"role": "user", "content": "say hi"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

`stream.text_stream` is a high-level helper that yields just the text deltas — exactly what the CLI tool needs to render a reply live. (Iterating the stream object directly gives you the raw event stream from the table above; we will need that for tool calls in Chapter 7, but `text_stream` covers everything in this chapter and Stretch Exercise 7 has you reproduce it from raw events.)

Our `llm` function in Chapter 3 returned a string — the full reply, fetched in one blocking call. We are going to change it to *yield* text deltas as they arrive, wrapping `messages.stream(...)` instead of `messages.create(...)`. That makes it a Python generator, and changes its type signature from `-> str` to `-> Iterator[str]`.

A *generator* is a function that uses `yield` instead of `return`. When you call it, you do not get its return value — you get a generator object that suspends the function's execution at each `yield` and resumes it on the next iteration. Local variables stay alive between yields, so the function effectively pauses mid-flight and picks up exactly where it left off when the caller asks for the next value. Generators are how Python expresses "produce values lazily, one at a time, without materialising the whole sequence in memory" — exactly what we want for a stream of text deltas. PEP 255 [3] is the original specification if you want the historical motivation.

Add the `Iterator` type to the imports at the top of `main.py`:

```python
from typing import Iterator
```

`Iterator` lives in the `typing` module [4]; built-in containers like `list` and `dict` work without an import, but parametrised types like `Iterator[str]` need it. Type hints are not enforced at runtime — they exist to be checked by tools and to communicate intent.

First of all, let's change the return type for the `llm()` function:

```python
def llm(messages: list[dict], system: str = "") -> Iterator[str]:
    """Stream the model's reply, yielding text deltas as they arrive."""
```
Now wrap `client.messages.stream(...)` in a `with` block and yield each delta as it arrives. The `with` matters here: when the generator is exhausted (or the caller breaks out early), the context manager closes the underlying HTTP connection cleanly. Forgetting it is a leak waiting to happen.

```python
def llm(messages: list[dict], system: str = "") -> Iterator[str]:
    """Stream the model's reply, yielding text deltas as they arrive."""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
```


Callers who want the full string get it via `"".join(llm(messages))`. Streaming is the new default; non-streaming becomes a one-liner over it. Nanobot makes a similar trade-off but keeps the two paths as separate methods — `chat_stream()` and `chat()` in `nanobot/nanobot/providers/anthropic_provider.py` — and the runner picks one based on whether the active hook (REPL, Telegram, WebUI) wants live deltas. Both share the same `_build_kwargs` and `_apply_cache_control` plumbing, so the request shape is identical; the only difference is whether the consumer renders text as it arrives.

With streaming, the SDK handles the work of pulling text out of the right content block. We do not care, here, that the model might also be emitting thinking blocks or tool blocks because `text_stream` filters to text.

## Updating chat to render live

Inside `chat()`, the single `reply = llm(messages, system=system)` call from Chapter 3 is replaced by a six-line block: print the assistant prefix, iterate the generator while printing each delta and collecting into a list, then append the joined reply to history.

```python
print("\nassistant: ", end="", flush=True)
chunks: list[str] = []
for text in llm(messages, system=system):
    print(text, end="", flush=True)
    chunks.append(text)
print("\n")
messages.append({"role": "assistant", "content": "".join(chunks)})
```

Two things in this block deserve a callout. The `flush=True` argument is mandatory: Python's `print` buffers stdout by default, and without flushing the "live" stream will not appear until the buffer fills or the program exits — the whole illusion collapses. And we accumulate deltas in a list and join at the end rather than building the reply with `reply += text`, because repeated string concatenation is technically O(n²) in Python. The reason is that strings are immutable: `reply += text` cannot grow the existing buffer in place, so the interpreter allocates a fresh string of length `len(reply) + len(text)`, copies the old contents into it, copies the new chunk on the end, and lets the old string be garbage-collected. Across N deltas of average size k, the total work is roughly `k + 2k + 3k + ... + Nk = O(N²·k)`, and the peak memory footprint includes both the old and new buffers at every step. A list of chunks plus one final `"".join(chunks)` is O(N) in time and O(N·k) in memory: the list holds N pointers to existing string objects (no copying), and `join` allocates the final buffer exactly once. The Python FAQ documents this idiom directly [5]. CPython has a special-case optimisation that sometimes mutates the buffer in place when the string has no other references, but it is fragile (any aliasing breaks it) and not something to rely on.


Run it:

```
$ uv run main.py
chat — Ctrl-D or empty line to exit

you: explain how a hash table works in 150 words

assistant: A hash table stores key-value pairs for fast lookup. It works in three steps:...
```

The reply now appears word by word instead of arriving as a single block.

## Getting the final message after streaming

Streaming gives us a stream of text. It does not, by itself, give us the things we got from the non-streaming response: `usage`, `stop_reason`, the full `content` list. Those are still available — the SDK accumulates them under the hood and exposes them via `stream.get_final_message()`, which you call after the stream has been consumed:

```python
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        ...
    final = stream.get_final_message()
    print(final.usage.input_tokens, final.usage.output_tokens)
    print(final.stop_reason)  # "end_turn", "max_tokens", "tool_use", ...
```

We are not threading this through `llm` yet. The current CLI tool does not need it, and adding it now means choosing how to expose final-message metadata across the generator boundary, which is a Chapter 7 question (when stop reasons start mattering for control flow). For now, if you want to inspect a final message, do it directly with `client.messages.stream` in a script. Exercises 3 and 4 push on this.

## Prompt caching

We now have a streaming client and a system prompt. Streaming reduced the time it takes to *display* a reply, but it did not change the time it takes to *start* generating one: every turn still re-processes the entire system prompt and conversation history from scratch. With the small example system prompt assembled from template files from Chapter 3 (a few hundred tokens of Markdown), nobody will notice. With real templates — many skill files, accumulated memory, instructions that grow over time — the system prompt will become the dominant cost on every turn.

Prompt caching is the lever for that. The Anthropic API charges roughly **10%** of the normal input-token rate (0.1×) for cached input, with a one-time write cost of about 1.25× when the cache is first populated [6]. Anthropic reports that long prompts can be served at up to 90% lower cost and 85% lower latency once a cache hit lands [7]. Mechanically, providers implement this by retaining the model's *KV cache* — the per-layer key/value tensors produced when the prefix was first processed — and reusing them on subsequent requests with the same prefix, skipping the prefill step entirely. The vLLM team's PagedAttention paper [8] is a good in-depth read on how serving systems manage KV-cache memory across many concurrent requests. Skipping prefill is also why cache reads improve time-to-first-token: there is no work to do before the first generated token comes out.

The mechanics on the Anthropic API are simple. You add `cache_control={"type": "ephemeral"}` at the top level of your request, and the API automatically caches the last cacheable block of the prefix:

```python
with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=system,
    cache_control={"type": "ephemeral"},  # <-- cache added
    messages=messages,
) as stream:
    ...
```

That is the entire change. The first request after a workspace edit *writes* the cache (at a 1.25× input rate). Subsequent requests within five minutes *read* the cache at the 10% rate. After five minutes of inactivity, the cache entry expires and the next request re-writes it [6]. A 1-hour TTL is also available at a higher 2× write cost, useful for prefixes that get reused across long sessions.

There are three things to know that are not obvious:

1. **Caching is a strict prefix match.** If you change *any* byte in the system prompt the cache is invalidated and you write fresh. Edit `persona.md` and the cache is gone until you rebuild it. This is fine for our use case (the workspace changes rarely) and a real footgun in production agents that interpolate timestamps into the system prompt. Chapter 24 spends time on the audit pattern for catching this kind of silent invalidation.

2. **There is a minimum cacheable size.** For Claude Opus 4.7 it is 4,096 tokens; smaller models have lower thresholds (Sonnet 4.6 sits at 2,048 tokens; Sonnet 4.5 at 1,024) [6]. If your prefix is shorter, the cache silently does nothing — `cache_creation_input_tokens` returns zero. Our example workspace is far below that threshold, so adding `cache_control` here is a no-op until the workspace grows. Adding the flag now is harmless and means the moment the workspace passes the threshold, we start saving money without further code changes.

3. **You can verify it is working.** The response's `usage` object reports `cache_creation_input_tokens` (tokens you paid 1.25× to write) and `cache_read_input_tokens` (tokens you paid 0.1× to read). After two consecutive requests with the same prefix, the second should report a non-zero `cache_read_input_tokens`. If it does not, something invalidated the cache between the calls — usually a silent timestamp or non-deterministic ordering somewhere in the prefix. Exercise 5 has you grow the templates directory until the cache kicks in and observe the difference.

## Production reference

Open `nanobot/nanobot/providers/anthropic_provider.py` and scroll to `chat_stream()` near the bottom. Strip away the surrounding error handling and you will see exactly what we just wrote: open `messages.stream(**kwargs)` as an async context manager, iterate `text_stream`, hand each delta to a callback, then call `get_final_message()` to recover `usage` and `stop_reason`. Three functions in particular map directly to what we built (or hand-waved) in this chapter:

- **`chat_stream()`** is the async equivalent of our generator `llm()`. Instead of `yield`, it pushes each delta into an `on_content_delta` callback supplied by the caller (the runner). Same shape, different concurrency model: a callback works better when one stream feeds several consumers (REPL, Telegram, observability) at the same time, which is what the runner needs.
- **`_apply_cache_control()`** is the production version of the single `cache_control={"type": "ephemeral"}` line we added. Notice it does more than mark the system prompt — it also walks the messages list and tools list and places markers at *each* of the four allowed cache breakpoints, so a tool catalog and a long conversation tail can be cached independently. The `_tool_cache_marker_indices` helper picks where the breakpoints go. Chapter 7 explains why the tool catalog is the second most expensive thing to cache.
- **`chat_stream_with_retry()`** in `nanobot/nanobot/providers/base.py` wraps `chat_stream` with retry-on-transient-failure logic. Streaming complicates retries — the first delta has already been emitted to the user when the connection drops mid-reply — and this is where the policy lives.

Two production nuances are worth flagging now, because they are invisible from a single-user CLI but inevitable the moment a real loop wraps the stream:

1. **Idle timeouts wrap the stream.** Each `__anext__()` on the iterator is wrapped in `asyncio.wait_for(..., timeout=idle_timeout_s)` (90 seconds by default, controlled by `NANOBOT_STREAM_IDLE_TIMEOUT_S`). Without this, a half-open TCP connection can hang the agent indefinitely; the SDK does not time out on its own.
2. **`get_final_message()` is treated as awaitable, not free.** It is wrapped in the same idle timeout. The final message arrives over the same connection as the deltas, and a network glitch at the end of a stream is just as fatal as one at the beginning.

Skim `chat_stream` once you have written your own version. The plain-text path is recognizably what we just wrote; the rest is bookkeeping for the things we will add in later chapters.

## Exercises

1. **Time to first token vs. total time.** Add timing around your `llm` generator: record the time at the start, the time when the first chunk arrives, and the time when the loop finishes. Print all three. Compare to the non-streaming version of `llm` (Chapter 3): the *total* time is roughly the same, but the *first-token* time is much shorter. That gap is the perceived-latency improvement.

2. **Disable `flush=True`.** Remove `flush=True` from the `print` call inside `chat`. Watch what happens. The stream is still arriving, but Python's stdout buffering hides it until the buffer fills or the program exits. This is a useful thing to have seen; it is also a footgun a lot of people hit at least once.

3. **Get the final message.** Modify `chat` to call `client.messages.stream(...)` directly (not via the `llm` generator) for one turn, so you can call `stream.get_final_message()` and inspect the result. Print `final.usage` and `final.stop_reason`. Confirm the `stop_reason` is `end_turn` for normal replies. We will need this in Chapters 7 and 9.

4. **Watch the cache.** Add `cache_control={"type": "ephemeral"}` to the stream call (it is already there in `main.py`). Modify `llm` (or the inline version from Exercise 3) to print `final.usage.cache_creation_input_tokens` and `final.usage.cache_read_input_tokens` after every turn. Run a 5-turn conversation and watch the numbers — most likely both will be zero, because the workspace is below the 4K minimum. That is expected; the next exercise fixes it.

5. **Force a cache hit.** Pad your workspace with a long Markdown file — a copy of an article, a code reference, anything that brings the system prompt above ~5,000 tokens. Re-run the conversation. The first turn should report a non-zero `cache_creation_input_tokens`; every subsequent turn (within five minutes) should report `cache_read_input_tokens` close to that number, with `input_tokens` (uncached) close to zero. You are now paying ~10% of the original system-prompt cost.

6. **Stretch: Cancel a stream.** Wrap the `for text in stream.text_stream` loop in a try/except for `KeyboardInterrupt`. When the user hits Ctrl-C, break out cleanly, append the partial reply to `messages`, and return to the input prompt. This is a small piece of "you can interrupt the agent" UX that real assistants need; we will treat it more carefully in Chapter 9.

7. **Stretch: Raw events.** Replace `stream.text_stream` with iteration over `stream` directly. Filter for `event.type == "content_block_delta"` and `event.delta.type == "text_delta"`. Confirm you get the same deltas that `text_stream` was giving you. Now also handle `event.type == "message_delta"` — print `event.delta.stop_reason` when it appears. You have just rebuilt `text_stream` plus a stop-reason hook from scratch; this is the path we will take in Chapter 7 for tool calls.

8. **Stretch: Stream without the SDK.** Drop down one more layer and call the API yourself. Use `httpx.stream("POST", "https://api.anthropic.com/v1/messages", json={..., "stream": True}, headers={"x-api-key": ..., "anthropic-version": "2023-06-01"})` and iterate `response.iter_lines()`. You will see raw SSE — `event: content_block_delta` lines paired with `data: {...}` JSON payloads, separated by blank lines. Parse them by hand: skip blanks, strip the `event: ` and `data: ` prefixes, `json.loads` the payload, dispatch on `type`, and reproduce the text-only output of `text_stream`. Compare your line count to `client.messages.stream`. This is what the SDK is doing for you on every call; once you have written it once you understand exactly why "just use the SDK" is the right default.

9. **Stretch: Cache audit.** Open `_apply_cache_control` in `nanobot/nanobot/providers/anthropic_provider.py` and notice that it places the `cache_control` marker on the *next-to-last* user message rather than the last one. Predict why before reading further. (Hint: think about what changes between consecutive turns and what does not, and which prefix you want to be cacheable on the *next* turn.) Then write a tiny audit helper in your own code that, given two consecutive `usage` objects, prints `cache_hit_ratio = cache_read / (cache_read + cache_creation + input_tokens)`. Run a 10-turn conversation with a padded workspace (per Exercise 5) and watch the ratio climb toward 1.0 after the second turn. Chapter 24 turns this into a proper observability widget.

10. **Stretch: Implement your own KV cache.** Prompt caching is KV-cache reuse with a pricing layer on top. This exercise builds it from the data structure up.

    *Part A — the data structure.* Write a `PrefixCache` class that supports:

    - `put(token_ids: list[int], state: Any) -> None` — store a state object at the end of a token sequence.
    - `get_longest_match(token_ids: list[int]) -> tuple[int, Any | None]` — return `(prefix_length, state)` for the deepest cached prefix of `token_ids`, or `(0, None)` if nothing matches.
    - `evict_older_than(seconds: int) -> None` — drop entries whose last access is past the TTL.

    The work breaks into four sub-tasks:

    1. **Pick the structure.** A flat dict keyed on `tuple(token_ids)` only matches *exact* prefixes — a 501-token prompt cannot reuse the cache from a 500-token prompt. A *trie* (one node per token, lookup walks one node per token, insertion shares structure across overlapping prompts) does longest-prefix lookup naturally. Each node holds one token, an optional state, and a `last_access_time`.

    2. **Decide what `put` does at intermediate nodes.** KV-cache state is decomposable — you can slice `past_key_values` to any prefix length — so storing the same state handle at every node along the put path is a defensible design and lets shorter queries reuse a longer cached prefix.

    3. **Choose a workload that exposes the structural win.** Many short queries that all begin with the same long system prompt is the realistic shape and the one where a trie crushes a flat dict: the trie shares the system prompt as one path, the dict stores N copies of it.

    4. **Profile both.** The trie should be linear in *unique* tokens (one shared path plus per-query suffixes); the naive dict linear in N · L. Time them against each other on the workload from sub-task 3.

    This is roughly the data structure a real serving system maintains, minus the GPU-paged-memory scheme that PagedAttention [8] adds on top.

    *Part B — wire it to a real model.* Install `torch` and `transformers` and load a small model locally — `distilgpt2` is around 350 MB and fine for this. First, time `model.generate(inputs, use_cache=False)` against `use_cache=True` on the same long prompt; the cached version should be markedly faster, and that gap is the prefill cost the provider's cache lets you skip. Then drive the model by hand instead of using `generate`: run one forward pass on a prefix of token IDs, capture `outputs.past_key_values` (a tuple of `(key, value)` tensors, one per transformer layer), and `put()` it into your `PrefixCache` from Part A. On the next call with a longer prompt that starts with the same prefix, `get_longest_match()` to retrieve the stored `past_key_values`, pass them back into the forward pass via the `past_key_values=` kwarg, and only feed the *new* tokens through the model. You have just implemented the trick a provider runs across millions of requests. The 5-minute TTL on Anthropic's cache exists for the same reason your trie eventually needs eviction: KV state is large, and you cannot keep every prefix forever.


## References

[1] *Using server-sent events.* MDN Web Docs. <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events>

[2] *Streaming Messages.* Claude API documentation. <https://platform.claude.com/docs/en/build-with-claude/streaming>

[3] *PEP 255 — Simple Generators.* <https://peps.python.org/pep-0255/>

[4] *typing — Support for type hints.* Python documentation. <https://docs.python.org/3/library/typing.html>

[5] *What is the most efficient way to concatenate many strings together?* Python FAQ. <https://docs.python.org/3/faq/programming.html#what-is-the-most-efficient-way-to-concatenate-many-strings-together>

[6] *Prompt caching.* Claude API documentation. <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>

[7] *Prompt caching with Claude.* Anthropic. <https://claude.com/blog/prompt-caching>

[8] Kwon, Woosuk, et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* arXiv:2309.06180. <https://arxiv.org/abs/2309.06180>