# Chapter 5: Provider Abstraction

We have spent four chapters writing against the Anthropic API. The `chat()` function in `main.py` already streams replies, threads context through the workspace, and prompt-caches the system prompt — but every byte still goes to one company. By the end of this chapter, the `llm()` function is gone, replaced by a `Provider` object that hides which API actually answers. We will write the abstraction itself, two concrete implementations covering most of the LLM ecosystem, and a `FallbackProvider` that keeps the agent alive when one of those APIs goes down.

## Why abstract at all

Four reasons cover most of the agent-engineering literature.

**Cost.** Frontier models trade leadership month-to-month, but the price gap between the top of the catalog and the small-model tier stays roughly an order of magnitude. As of 2026, Claude Opus 4.7 charges $15 per million input tokens and $75 per million output tokens [1]; Claude Haiku 4.5 sits at $1 / $5; OpenAI's `gpt-5-mini` is in the same neighborhood as Haiku; Gemini 2.5 Flash-Lite charges $0.10 / $0.40 [2]; a Llama 3 model running locally on a workstation GPU is effectively free per token (you pay in electricity and capital). Most agent turns — picking which tool to call, summarizing a tool result, deciding to ask a clarifying question — do not need the most expensive model on the menu. Chapter 19 will explicitly route subagents to cheaper models; the abstraction in this chapter is what lets that routing exist at all.

**Capability.** The frontier is currently a plateau. On the LMArena text leaderboard, the top three models sit within overlapping 95% confidence intervals — a statistical tie that rotates week to week [3]. Aggregate "intelligence" indices like Artificial Analysis's [4] tell the same story: Claude, Gemini, and GPT cluster within a few points across the suite. The real spreads show up on specific benchmarks: Claude leads coding-style work on SWE-bench Verified by a meaningful margin [5], Gemini wins on cost-per-token at the small-model tier, OpenAI's models tend to lead on multimodal eval rounds. The benchmarks themselves move quickly — older ones like MMLU-Pro are saturating at the top, and GPQA Diamond and HLE are the current frontier discriminators [6]. An agent that can talk to all of them has more to draw on, *and* can be re-pointed at the new leader the week the leaderboard shifts. Both Artificial Analysis [4] and the LMArena leaderboard [3] are reasonable places to track this in real time.

**Privacy.** Some inputs should never leave your machine. Patient records under HIPAA, source code that constitutes trade secrets, attorney-client correspondence under privilege, financial statements before public disclosure, draft research before submission — for any of these, "we send it to a third-party API" is a non-starter regardless of how good the model is. The way agents address this is by running an open-weights model — Llama [7], Mistral [8], Qwen [9] — locally through a runner like Ollama [10] or LM Studio [11]. Both runners expose an HTTP endpoint with the same shape as a hosted provider, so swapping a remote model for a local one becomes a configuration change at the provider layer rather than a code change anywhere else.

**Availability.** Hosted providers go down. Anthropic, OpenAI, and Google all maintain public status pages [12][13][14] and all of them post incidents on a regular cadence. An agent that can fall back from one provider to another keeps working through the outage; an agent wired to a single SDK stops at the first 503. The last section of this chapter implements that failover in about thirty lines.

The abstraction we build here makes all four of these tractable: cost routing, capability routing, on-prem deployment, and failover all become questions of which `Provider` instance you hand to `chat()`.

## What is actually different between providers

Before we abstract, it is worth seeing the differences concretely. Here is the same call against three SDKs, with the variations highlighted [15][16][17].

**Anthropic:**

```python
client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    system="You are concise.",        # separate parameter
    messages=[{"role": "user", "content": "Hi"}],
)
# response.content[0].text
```

**OpenAI:**

```python
client.chat.completions.create(
    model="gpt-5",
    max_completion_tokens=16000,
    messages=[
        {"role": "system", "content": "You are concise."},  # first message
        {"role": "user", "content": "Hi"},
    ],
)
# response.choices[0].message.content
```

**Google Gemini** (native SDK):

```python
client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Hi",                                  # not "messages"
    config=types.GenerateContentConfig(
        system_instruction="You are concise.",      # third name for the same thing
    ),
)
# response.text
```

The system prompt has a different name in each SDK, so does the message list, and the response is unwrapped from a different field. Underneath, the structure is identical: system instruction, ordered user/assistant turns, model identifier, generated text. That common structure is what the `Provider` interface lifts out.

## The convergence on OpenAI-compatible APIs

In practice, you do not need a separate adapter per provider. Some time around 2024, "OpenAI-compatible" became the de facto interface for non-Anthropic LLM endpoints [18][19]. By 2026, Google Gemini, every local-model runner (Ollama, LM Studio, vLLM, llama.cpp), every aggregator (OpenRouter, Together, Replicate), and most cloud-hosted open-weights providers (Mistral, DeepSeek, Qwen, Kimi) all expose an HTTP endpoint that speaks OpenAI's request and response shape. You point the official `openai` Python SDK at their `base_url` with their `api_key`, and everything works.

Gemini is the example most relevant to this chapter. The native shape was the third snippet above; through the OpenAI SDK it becomes:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
client.chat.completions.create(
    model="gemini-2.5-flash-lite",
    messages=[{"role": "user", "content": "Hi"}],
)
```

The model identifier rotates fast — Google deprecated `gemini-2.5-flash` in mid-2026 — so always check the current Gemini docs [20] before wiring a specific id into a long-lived config.

Anthropic also offers an OpenAI-compatible endpoint [21], but the documentation is unusually direct about its scope: it is "primarily intended to test and compare model capabilities, and is not considered a long-term or production-ready solution." The unsupported list is the reason. **Prompt caching is not supported** — every request pays the full system-prompt cost, which is what made the entire Chapter 4 effort worthwhile. Citations, structured outputs (`strict` is silently ignored), and PDF processing all fall back to text-only behavior. Extended thinking works as a black box: the model thinks, but you cannot read the thinking deltas. Cache statistics in the response are always empty. The pattern is consistent — features that fit the OpenAI request shape are translated faithfully; features Anthropic ships that do not have an OpenAI counterpart are dropped silently. That is a useful thing to have seen concretely once: it is exactly the leaky-abstraction problem we will revisit in the litellm section. For Anthropic specifically, the right answer is to keep the native SDK in `AnthropicProvider` and not route Claude through an OpenAI client.

So we need exactly two concrete implementations: one for Anthropic (native), one for any OpenAI-compatible endpoint (covering OpenAI itself, Gemini, OpenRouter, Ollama, LM Studio, and everything else).

## The `Provider` interface

The interface should hide which HTTP call we are making and expose only what `chat()` needs. Looking at the existing `llm()` function, the signature falls out for free: take a list of messages and a system prompt, yield text deltas. Create a `providers/` directory in `agent/` and add `base.py`:

```python
from abc import ABC, abstractmethod
from typing import Iterator


class Provider(ABC):
    """A streaming LLM backend."""

    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """Stream the model's reply, yielding text deltas as they arrive."""
        ...
```

Some of the Python here is worth a closer look, especially if you come from Java or C++.

`ABC` is the *abstract base class* marker from the `abc` module [22]. A class that inherits from `ABC` and contains at least one `@abstractmethod` cannot be instantiated directly — Python raises `TypeError` if you try. This is Python's way of declaring an interface. Java would use the `interface` keyword and `abstract` methods; C++ uses pure virtual methods (`virtual void f() = 0;`) on a class with no implementation. The ergonomics differ — Python has no compile step, so the check happens the first time you try to construct the class — but the contract is the same: subclasses must override every `@abstractmethod` to be instantiable.

Inside `stream`, the body is just `...` — the *Ellipsis* literal. The body is never executed for abstract methods, so a one-character placeholder is conventional. `pass` would also work; `...` is preferred in modern Python because it visually signals "intentionally empty."

The `Iterator[str]` return type comes from `typing` (Python 3.9+ also accepts `collections.abc.Iterator`). It says "this returns something you can iterate over, and each iteration yields a `str`." Generators (functions with `yield`) automatically satisfy this; so do plain iterators.

A few words on the interface design itself. The provider takes its configuration once, at construction time — model name, max tokens, API key, base URL — and exposes only `stream()`. That means caching policies, custom headers, timeout values, and SDK-specific kwargs all stay inside the provider. The agent never sees them. If we had instead passed those things into `stream()` on every call, the abstraction would leak in both directions: the agent would have to know about Anthropic's `cache_control` argument, and the provider would have no way to enforce its own defaults.

## Implementing `AnthropicProvider`

In the `providers/` folder, create `anthropic_provider.py`. The implementation is the existing `llm()` function wrapped into a class — `model` and `max_tokens` become construction-time parameters instead of hard-coded defaults:

A note on the import: write `from providers.base import Provider`, not `from base import Provider`. The bare `from base import …` form only works if the file you are running happens to be inside `providers/`; running `uv run main.py` from `agent/` will fail with `ModuleNotFoundError: No module named 'base'`. The package-qualified form works regardless of which directory the entry point lives in, and it matches how `main.py` already imports the same symbols.

```python
from typing import Iterator

import anthropic

from providers.base import Provider


class AnthropicProvider(Provider):
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 16000,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            cache_control={"type": "ephemeral"},
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
```

The `cache_control={"type": "ephemeral"}` flag from Chapter 4 stays here — it is Anthropic-specific and there is no reason to abstract it. OpenAI runs an automatic cache for prompts above a length threshold and requires no caller-side configuration [23], so the OpenAI provider will simply not pass anything cache-related. The `Provider` interface stays clean; each implementation handles caching the way its API expects.

## Implementing `OpenAIProvider`

Add the dependency:

```bash
uv add openai
```

This updates `pyproject.toml` and `uv.lock`. From now on `uv run main.py` will see the `openai` package without any additional steps.

Now create `openai_compatible_provider.py`. The provider has two new ideas — *translating* between the two message-shape conventions, and *iterating* OpenAI's stream events instead of Anthropic's — so we will introduce them one at a time. Start with construction and translation. No streaming yet:

```python
from typing import Iterator

from openai import OpenAI

from providers.base import Provider


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int = 16000,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=oai_messages,
        )
        yield response.choices[0].message.content
```

Compared to the Anthropic version, the differences are small:

- The system prompt goes into the messages list as the first entry with `role="system"`, instead of being a separate parameter. That is the *translation*. The two-line `if system: ...append(...)` then `extend(messages)` is the entire translation logic.
- The token-budget parameter is `max_completion_tokens`, not `max_tokens`. OpenAI renamed it for newer models; Anthropic stayed with the original name. The provider abstracts the difference away so callers do not care.

This works, but it does not stream — the function yields the entire reply in one chunk after the API call finishes. To switch to real streaming, change two lines:

```python
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=oai_messages,
            stream=True,                                   # <-- was missing
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content      # <-- was a single yield
```

A few streaming-specific things to notice:

- **No `with` block.** OpenAI's streaming response is a plain iterator, not a context manager. Anthropic wraps the stream in `with` so it can clean up the underlying HTTP connection deterministically; OpenAI's SDK relies on garbage collection. Both work; the shapes are just different. This is a real ergonomic difference and a small reason to use both SDKs through an abstraction layer.
- **`chunk.choices[0].delta.content` can be empty.** OpenAI's stream events are nested inside a list of choices, and a delta with no `content` is a signal that the chunk is something else — usually a tool-call delta (Chapter 7) or a finish-reason marker. The `if chunk.choices and chunk.choices[0].delta.content` guard is what filters down to actual text. We will revisit the empty-content branches when we add tool calls.

Because the abstraction is "anything that speaks OpenAI's wire format," the same `OpenAIProvider` class talks to many endpoints just by changing `base_url`:

```python
import os

# OpenAI proper
provider = OpenAIProvider(model="gpt-5")

# Gemini, via OpenAI-compatible endpoint
provider = OpenAIProvider(
    model="gemini-2.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# A local Llama running in Ollama
provider = OpenAIProvider(
    model="llama3.1",
    api_key="ollama",  # required by the SDK; Ollama ignores it
    base_url="http://localhost:11434/v1",
)

# OpenRouter, fronting hundreds of models
provider = OpenAIProvider(
    model="anthropic/claude-sonnet-4-6",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

For Gemini specifically, the OpenAI-compat endpoint is the practical default. If you find yourself wanting a Gemini-only feature — tuned models, file-based grounding, Vertex AI — you can write a third provider that uses Google's native SDK; Exercise 1 walks through it.

## Wiring it up

In `main.py`, the `llm()` function is now obsolete — its body lives inside `AnthropicProvider.stream()`. Delete the function, delete the `import anthropic` and the module-level `client` it relied on, and replace them with a default provider:

```python
from providers.anthropic_provider import AnthropicProvider
from providers.base import Provider

DEFAULT_PROVIDER = AnthropicProvider()
```

`chat()` takes a provider argument, falling back to the default:

```python
def chat(provider: Provider | None = None) -> None:
    """Run an interactive chat loop, accumulating turns in a single messages list."""
    if provider is None:
        provider = DEFAULT_PROVIDER
    # ... rest of the function ...
```

And the streaming loop calls the provider instead of the deleted `llm()`:

```python
for text in provider.stream(messages, system=system):
    print(text, end="", flush=True)
    chunks.append(text)
```

`chat()` no longer mentions Anthropic anywhere. To switch the whole agent to Gemini or to a local model, change the one-line `DEFAULT_PROVIDER = ...` assignment.

## Verifying the swap actually works

The previous paragraph is a promise. Before building anything else on top of the `Provider` interface, let's cash that promise in and run the same agent against a different backend.

Get a Gemini API key from Google AI Studio (free tier is enough for this) and add it to your `.env`:

```
GEMINI_API_KEY=...
```

Then change the one-line default in `main.py` to point at Gemini through the OpenAI-compatible endpoint:

```python
import os

from providers.openai_compatible_provider import OpenAIProvider

DEFAULT_PROVIDER = OpenAIProvider(
    model="gemini-2.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
```

Run `uv run main.py` and ask the model who it is:

```
chat — Ctrl-D or empty line to exit

you: who made you, and what model are you running?
assistant: I'm Gemini, a large language model from Google.
```

Switch the line back to `AnthropicProvider()` and ask the same question:

```
you: who made you, and what model are you running?
assistant: I'm Claude, made by Anthropic.
```

Same `chat()` loop, same workspace context, same streaming, same conversation-history accumulation — different backend, different model, different voice. Nothing outside the `DEFAULT_PROVIDER` assignment changed. The Anthropic-specific `cache_control` flag is no longer in the picture when Gemini answers, because `OpenAIProvider.stream` simply does not pass it; the cache markers stay sealed inside `AnthropicProvider` where they belong.

The same one-line trick covers OpenAI itself (`OpenAIProvider(model="gpt-5")` with `OPENAI_API_KEY` in your `.env`) and a local model in Ollama (`OpenAIProvider(model="llama3.1", api_key="ollama", base_url="http://localhost:11434/v1")` — no key needed). The next section makes the swap automatic: instead of you editing the line when one backend is down, the agent picks the next one on its own.

## Building automatic failover

Now the payoff. The whole point of having a `Provider` interface is that a higher-level provider can wrap others. A `FallbackProvider` takes an ordered list of backends and tries them in turn — if the first one raises a transient error, it moves on to the second.

Create `providers/fallback_provider.py`:

```python
from typing import Iterator

from providers.base import Provider


class FallbackProvider(Provider):
    def __init__(self, providers: list[Provider]):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                yielded_anything = False
                for text in provider.stream(messages, system=system):
                    yielded_anything = True
                    yield text
                return
            except Exception as e:
                if yielded_anything:
                    raise
                last_error = e
                continue
        raise RuntimeError(
            f"All providers failed; last error: {last_error!r}"
        )
```

Both subtleties in those twenty lines come down to *when* it is safe to fall back.

The `yielded_anything` flag is the load-bearing one. If the first provider has already streamed three sentences to the user before the connection drops, falling back to a second provider would mean the user sees those three sentences continue with output from a *different* model in the same paragraph. That is worse than failing — the message becomes a Frankenstein and the agent's context now has a half-reply from one model in `messages`. So once the first delta has been emitted, an exception terminates the whole call instead of triggering a fallback. Pre-flight failures (auth errors, rate limits, connection errors that fire before any byte is received) are the only cases that legitimately fall back.

The other subtlety is what "Exception" means here. Catching everything is the simple version and is fine for a chapter. In production, you want to distinguish *retryable* errors (HTTP 429 rate-limit, HTTP 5xx, connection timeouts) from *non-retryable* ones (HTTP 400 bad request, HTTP 401 auth) — the second class will fail on the next provider too, and trying it just adds latency to a guaranteed failure. Exercise 2 asks you to add that distinction.

To use it, replace the `DEFAULT_PROVIDER` line in `main.py`:

```python
import os

from providers.anthropic_provider import AnthropicProvider
from providers.fallback_provider import FallbackProvider
from providers.openai_compatible_provider import OpenAIProvider

DEFAULT_PROVIDER = FallbackProvider([
    AnthropicProvider(),
    OpenAIProvider(model="gpt-5"),
    OpenAIProvider(
        model="gemini-2.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
])
```

The agent now keeps working through an Anthropic outage, an OpenAI outage, *and* a Gemini outage — it only stops when all three provider APIs are down at the same time.

One practical detail before testing: the snippet above assumes you have all three API keys set. The `OpenAI()` and `anthropic.Anthropic()` constructors raise immediately if a key is missing, so a single missing variable will crash `main.py` at import time. If you only have one or two of the keys, build the chain conditionally instead:

```python
chain: list[Provider] = []
if os.environ.get("ANTHROPIC_API_KEY"):
    chain.append(AnthropicProvider())
if os.environ.get("OPENAI_API_KEY"):
    chain.append(OpenAIProvider(model="gpt-5"))
if os.environ.get("GEMINI_API_KEY"):
    chain.append(OpenAIProvider(
        model="gemini-2.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ))
DEFAULT_PROVIDER = FallbackProvider(chain)
```

The agent code in the repo for this chapter uses this conditional form so a reader with only an Anthropic key still gets a working agent.

### Testing the failover

You do not need an actual outage to confirm the fallback works — overriding an environment variable on a single command is enough. Each of the following invocations runs `main.py` with one or more API keys deliberately replaced by a bogus value, and you can observe which provider answered by asking the model who it is.

The order matters. The agent tries providers in the order they appear in the chain, so the first provider with a *valid* key wins.

**1. Healthy run (baseline).** All real keys, request goes to Anthropic — first in the chain:

```bash
uv run main.py
# you: who made you?
# assistant: I'm Claude, made by Anthropic.
```

**2. Anthropic down → OpenAI takes over.**

```bash
ANTHROPIC_API_KEY=bogus uv run main.py
# you: who made you?
# assistant: I'm ChatGPT, a large language model from OpenAI.
```

The first try (Anthropic) raises an authentication error immediately, before any byte has been streamed, so `FallbackProvider` walks to the next entry. The user sees no error — only a slightly higher first-token latency.

**3. Anthropic and OpenAI down → Gemini takes over.**

```bash
ANTHROPIC_API_KEY=bogus OPENAI_API_KEY=bogus uv run main.py
# you: who made you?
# assistant: I'm Gemini, from Google.
```

**4. All three down → the chain bottoms out.**

```bash
ANTHROPIC_API_KEY=bogus OPENAI_API_KEY=bogus GEMINI_API_KEY=bogus uv run main.py
# you: who made you?
# RuntimeError: All providers failed; last error: AuthenticationError(...)
```

This is the only path that surfaces a user-facing error. Up to that point the abstraction has been silent — exactly what you want from failover.

What this exercise *does not* verify is the mid-stream guard (`yielded_anything`). Reproducing that requires either a real network drop in the middle of a reply or a more elaborate fault-injection rig. Exercise 2 walks through the production-grade error handling, which is where the mid-stream behavior becomes load-bearing.

## A note on `litellm` and similar libraries

If you have looked at the agent ecosystem, you have probably seen `litellm` [24] — a Python library whose entire job is to be the abstraction we just wrote, but for a hundred providers instead of two. You can use it, and many production agents do.

We chose not to, for two reasons.

First, the book's purpose is to understand how agents work. A library that hides every provider under one function call does not teach you the foundational concepts. After this chapter you can read `litellm`'s source code and recognize what it is doing — Exercise 4 walks through that mapping.

Second, and more important: in the real world, every provider library leaks. The leaks happen most often around *tool calls*, especially streaming tool calls (Chapter 7). The reason is structural. OpenAI streams a tool call by accumulating a JSON-string `arguments` field across many delta chunks — the model emits `{"loc` then `ation":"S` then `F"}` and the SDK reassembles. Anthropic streams the equivalent as a `tool_use` content block with typed `input_json_delta` events, then a `message_delta` with the final `stop_reason`. A library that wants to look like OpenAI to its callers has to either translate Anthropic's typed deltas into a fake JSON-string accumulator (losing the type information that made Anthropic's stream useful) or expose a leaky union type that callers have to unpack themselves. Both happen, and both are sources of bugs that production agents eventually hit and have to patch around.

Nanobot specifically dropped `litellm` in March 2026, in commit `3dfdab7` [25]: "Remove litellm dependency entirely (supply chain risk mitigation) … 593 tests passed, net -1034 lines." The supply-chain note matters — earlier commit `38ce054` had pinned the version after a security advisory — but the line count is the structural story. Replacing a 100-provider library with two native SDKs *removed* a thousand lines of code, because almost all of the litellm-related code was workarounds for the abstraction's leaks.

The proposal that led to that commit [26] is worth reading in full if you are picking between a meta-library and rolling your own provider layer. Three threads of argument run through the discussion.

**Dependency weight versus project philosophy.** Nanobot is around 4,000 lines of code total — its identity is "ultra-lightweight." Pulling in a roughly 30 MB dependency that supports 126 providers most users never touch is structurally heavier than the project itself. The three native SDKs nanobot actually needs (Anthropic, OpenAI, Google) weigh about 5 MB combined. The 6× size delta is not the headline on its own; the headline is the misalignment between a heavyweight abstraction and a lightweight project. If your project is large to begin with, this argument has less weight; if your project is small, an outsized dependency is a smell.

**Transparency and debuggability.** With native SDKs, errors come out of the SDK directly and map to documented HTTP responses you can look up. With `litellm` in the path, users were hitting Pydantic serialization warnings that did not correspond to anything the underlying provider had returned — by the time the warning surfaced, the abstraction had already transformed both the request and the response, and the user had no leverage on what to fix. "Fully visible" code, in the proposal's phrasing, beats a black box that mediates the contract.

**Provider-specific feature access.** This is the same leak we already covered from the other direction (the Anthropic-via-OpenAI-SDK section earlier in the chapter): when a translation layer normalizes everything to the lowest common denominator, provider-specific features — extended thinking, prompt caching, citations — end up either dropped or grafted on as extensions that defeat the abstraction's purpose. Talking to each SDK directly is what keeps those features first-class. For a chapter that just spent its entire previous installment caching the system prompt, this argument is load-bearing.

The acknowledged tradeoff is provider coverage: `litellm`'s 126 providers down to roughly 50 between native-SDK and OpenAI-compatible support. In practice OpenRouter mitigates most of this, since it fronts hundreds of long-tail models behind a single OpenAI-compatible endpoint that the `OpenAIProvider` you wrote already speaks. The cost of going native is real but bounded.

## Production reference

Open [`nanobot/nanobot/providers/`](https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers). The structure is recognizably what we just built, with a layer of factory and registry on top:

- **[`base.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py)** — the abstract `Provider` (named [`LLMProvider`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L91) in nanobot, recognizably the same idea).
- **[`registry.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/registry.py)** and **[`factory.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/factory.py)** — wire up provider construction from configuration files. Our `DEFAULT_PROVIDER = ...` line is the simplest possible version of this; in production you want to declare providers in YAML and let the factory build them.
- **[`anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py)** — Anthropic's API. The [`_apply_cache_control()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L379) method is the production version of our single `cache_control={"type": "ephemeral"}` line. It walks the system, messages, and tools lists and places markers at *each* of the four allowed cache breakpoints, so a long tool catalog and a long conversation tail can be cached independently.
- **[`openai_compat_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py)** — the catch-all for OpenAI-format endpoints. This single file covers Gemini-via-compat, OpenRouter, Ollama, vLLM, LM Studio, DeepSeek, Mistral, Kimi, and several specific cloud models. Skim its [`chat_stream`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py#L1199) method — the shape is what we wrote, plus extra handling for tool-call deltas (Chapter 7), cancellation (Chapter 9), and structured event emission (Chapter 22).
- **[`azure_openai_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/azure_openai_provider.py)** and **[`github_copilot_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/github_copilot_provider.py)** — variants of OpenAI-compatible with provider-specific authentication.
- **[`openai_responses/`](https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_responses)** — the newer Responses API path; the book does not cover it.

A couple of production nuances are visible in those files but invisible from our toy:

**Async, with one client per provider.** Nanobot uses `AsyncAnthropic` and `AsyncOpenAI`, with `max_retries=0` on the SDK and a centralized retry policy in [`chat_stream_with_retry()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L527) (in [`base.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py)). The reason for `max_retries=0` is that the SDK's default exponential-backoff retry interacts badly with streaming — retrying a half-delivered stream produces duplicate output. Centralizing retries lets the policy be aware of which streams are mid-flight.

**Idle timeouts on every event.** Each `__anext__()` on the stream is wrapped in `asyncio.wait_for(..., timeout=idle_timeout_s)` (90 seconds by default, controlled by `NANOBOT_STREAM_IDLE_TIMEOUT_S`). A half-open TCP connection can hang the agent indefinitely otherwise — the SDK does not time out on its own.

## Exercises

1. **Native Gemini provider.** Install `google-genai` and write a `GeminiProvider` that uses Gemini's native SDK (the third example near the top of this chapter). It will be more fiddly than `OpenAIProvider` because the message shape is more different — `contents` instead of `messages`, `system_instruction` inside a config object, a different stream API. Once it works, decide for yourself whether you would rather have it or use Gemini through `OpenAIProvider`. The right answer depends on whether you need any Gemini-specific feature.

2. **Smarter `FallbackProvider`.** Extend the version from this chapter to distinguish retryable from non-retryable errors. Catch the SDK exceptions explicitly: `anthropic.RateLimitError`, `anthropic.APIStatusError`, `openai.RateLimitError`, `openai.APIStatusError`, `httpx.ConnectError`. Fall back on rate limits, 5xx status codes, and connection errors; re-raise everything else (auth errors, 4xx bad-request errors) immediately, since the next provider will fail the same way. Add exponential backoff between attempts. Test it by setting one provider's API key to a bogus value (auth failure → re-raise immediately, no fallback) and a different one to a real key (works on first try).

3. **Cost-routed provider.** Write a `RoutedProvider` that picks a cheaper backend for short messages and a stronger one for longer messages. Trivial heuristic: if `sum(len(m["content"]) for m in messages) < 2000`, use Haiku or `gpt-5-mini`; otherwise use Opus or `gpt-5`. This is a tiny example of the routing pattern Chapter 19 will revisit when subagents need to choose their own backend.

4. **Stretch: map `litellm` to your provider abstraction.** Clone `litellm` (the repository for this book already has it under [litellm/](../litellm/)) and open `litellm/llms/base_llm/chat/transformation.py`. Read `BaseConfig` carefully — focus on `transform_request`, `map_openai_params`, and `get_supported_openai_params`. Then open `litellm/llms/anthropic/chat/transformation.py` and read `AnthropicConfig.transform_request`. Notice the direction of translation: in litellm, the *caller* always speaks OpenAI's shape, and each `transform_request` translates *outward* to the provider's native shape. Now refactor your own `AnthropicProvider` to accept the same input convention — a flat OpenAI-style `messages` list where the system prompt is `messages[0]` with `role="system"`, instead of the current `(messages, system)` split. You will need a small `_split_system(messages)` helper that pulls out the system message and reformats. Verify your existing `chat()` loop still works after the change. Then write a paragraph on which direction is easier to teach (caller-shape-as-OpenAI vs. caller-shape-as-our-`Provider`) and which is easier to scale to a hundred providers.

5. **Stretch: nanobot's prompt-caching strategy.** Open `nanobot/nanobot/providers/anthropic_provider.py` and find `_apply_cache_control` (around line 379). Notice it places markers in three places: the tail of the system prompt, `messages[-2]`, and indexed tool entries. Implement a simplified version inside your own `AnthropicProvider.stream`: take `system` and `messages`, and attach `cache_control={"type": "ephemeral"}` to the system message *and* to `messages[-2]` (skip the second marker if there are fewer than three messages). Run a 6-turn conversation against a padded workspace (use the technique from Chapter 4 Exercise 5 to push the system prompt above the cache threshold), and print `final.usage.cache_read_input_tokens` after each turn. The numbers should climb starting at turn 3. Then predict why nanobot caches `messages[-2]` and not `messages[-1]`, validate your prediction by trying both, and compare the read counts. Hint: think about what changes between consecutive turns and what does not.

6. **Stretch: nanobot's error categorization.** Open `nanobot/nanobot/providers/openai_compat_provider.py` and find where it categorizes errors. Notice how it distinguishes rate-limit errors from context-window-exceeded errors from generic transient errors. Add the same categorization to your own `OpenAIProvider`: define a small enum (`ErrorCategory.RATELIMIT`, `ErrorCategory.CONTEXT_WINDOW`, `ErrorCategory.TRANSIENT`, `ErrorCategory.PERMANENT`) and a helper `_categorize(error: Exception) -> ErrorCategory`. Wire the categorization into your `FallbackProvider` from this chapter so that `CONTEXT_WINDOW` errors are re-raised immediately (the next provider has the same context limit), `RATELIMIT` and `TRANSIENT` trigger a fallback, and `PERMANENT` errors are re-raised. Test each branch.

## References

[1] *Pricing.* Anthropic. <https://www.anthropic.com/pricing>

[2] *Gemini 2.5 Flash-Lite is now stable and generally available.* Google Developers Blog. <https://developers.googleblog.com/en/gemini-25-flash-lite-is-now-stable-and-generally-available/>

[3] *LMArena leaderboard.* <https://lmarena.ai/>

[4] *Artificial Analysis: AI model and API provider comparison.* <https://artificialanalysis.ai/>

[5] *SWE-bench Verified leaderboard.* Artificial Analysis. <https://artificialanalysis.ai/evaluations/swe-bench-verified>

[6] *GPQA Diamond benchmark.* Artificial Analysis. <https://artificialanalysis.ai/evaluations/gpqa-diamond>

[7] *Llama.* Meta. <https://www.llama.com/>

[8] *Mistral AI.* <https://mistral.ai/>

[9] *Qwen.* Alibaba. <https://qwen.ai/>

[10] *Ollama.* <https://ollama.com/>

[11] *LM Studio.* <https://lmstudio.ai/>

[12] *Anthropic status.* <https://status.anthropic.com/>

[13] *OpenAI status.* <https://status.openai.com/>

[14] *Google Cloud status dashboard.* <https://status.cloud.google.com/>

[15] *Messages.* Claude API documentation. <https://platform.claude.com/docs/en/api/messages>

[16] *Chat completions.* OpenAI API reference. <https://platform.openai.com/docs/api-reference/chat>

[17] *Generate content with the Gemini API.* Google AI for Developers. <https://ai.google.dev/gemini-api/docs/text-generation>

[18] Tony Lixu. *OpenAI API: The De Facto Standard for LLM Programming.* Medium. <https://tonylixu.medium.com/openai-api-the-de-facto-standard-for-llm-programming-part-one-484393f7161a>

[19] *OpenAI-compatible API.* BentoML LLM Inference Handbook. <https://bentoml.com/llm/llm-inference-basics/openai-compatible-api>

[20] *Gemini API: Models.* Google AI for Developers. <https://ai.google.dev/gemini-api/docs/models>

[21] *OpenAI SDK compatibility.* Claude API documentation. <https://platform.claude.com/docs/en/api/openai-sdk>

[22] *abc — Abstract Base Classes.* Python documentation. <https://docs.python.org/3/library/abc.html>

[23] *Prompt caching.* OpenAI documentation. <https://platform.openai.com/docs/guides/prompt-caching>

[24] *BerriAI/litellm.* GitHub. <https://github.com/BerriAI/litellm>

[25] Xubin Ren. *refactor: replace litellm with native openai + anthropic SDKs.* Nanobot commit `3dfdab7`, 2026-03-24. <https://github.com/HKUDS/nanobot/commit/3dfdab704e14b99de3ac93b24642eb9f09daab44>

[26] *Proposal: Replace litellm with native OpenAI + Anthropic SDKs.* Nanobot issue #161. <https://github.com/HKUDS/nanobot/issues/161>
