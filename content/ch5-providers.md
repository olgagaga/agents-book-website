# Chapter 5: Provider Abstraction

Four chapters so far have been written against the Anthropic API. The `chat()` function in `main.py` already streams replies, threads context through the workspace, and prompt-caches the system prompt. 

By the end of this chapter, the `llm()` function is gone, replaced by a `Provider` object that hides which API actually answers. This chapter builds the abstraction itself, two concrete implementations covering most of the LLM ecosystem, and a `FallbackProvider` that keeps the agent alive when one of those APIs goes down.

## Why abstract at all

Frontier models trade leadership month-to-month, but the price gap between the top of the catalog and the small-model tier stays roughly an order of magnitude. As of 2026, Claude Opus 4.7 charges $15 per million input tokens and $75 per million output tokens [1]; Claude Haiku 4.5 sits at $1 / $5; OpenAI's `gpt-5-mini` is in the same neighborhood as Haiku; Gemini 2.5 Flash-Lite charges $0.10 / $0.40 [2]; a Llama 3 model running locally on a workstation GPU is effectively free per token (the cost is electricity and capital). Most agent turns include picking which tool to call, summarizing a tool result, and deciding to ask a clarifying question. They often do not need the most expensive model. Chapter 19 will explicitly route subagents to cheaper models with the help of the abstraction defined in this chapter.

At the same time, on the LMArena text leaderboard, the top three models sit within overlapping 95% confidence intervals. This is a statistical tie that rotates week to week [3]. Aggregate "intelligence" indices like Artificial Analysis's [4] tell the same story: Claude, Gemini, and GPT cluster within a few points across the suite. The real spreads show up on specific benchmarks. Claude, for instance, leads coding-style work on SWE-bench Verified by a meaningful margin [5], while at the small-model tier the deciding factor is often cost per token rather than capability, where Gemini's Flash-Lite pricing is among the lowest of the hosted options [2]. The benchmarks themselves move quickly: older ones saturate at the top and stop discriminating between frontier models, while harder, newer ones like GPQA Diamond take over as the yardstick [6]. An agent that can talk to all of them has more to draw on and can be re-pointed at the new leader the week the leaderboard shifts. Both Artificial Analysis [4] and the LMArena leaderboard [3] are reasonable places to track this in real time.

Another concern worth mentioning to support the abstraction layer is privacy. Some inputs should never leave the machine. Patient records under HIPAA, source code that constitutes trade secrets, attorney-client correspondence under privilege, financial statements before public disclosure, draft research before submission are good examples. The way agents address this is by running an open-weights model — Llama [7], Mistral [8], Qwen [9] — locally through a runner like Ollama [10] or LM Studio [11]. Both runners expose an HTTP endpoint with the same shape as a hosted provider, so swapping a remote model for a local one becomes a configuration change at the provider layer rather than a code change anywhere else.

Another thing is that hosted providers go down. Anthropic, OpenAI, and Google all maintain public status pages [12][13][14] and all of them post incidents on a regular cadence. An agent that can fall back from one provider to another keeps working through the outage compared to an agent tied to one specific SDK. 

The abstraction built here makes all four of these tractable.

## What is actually different between providers

Before abstracting, it is worth seeing the differences concretely. Below is the same call against three SDKs, with the variations highlighted [15][16][17].

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

Every SDK carries the same four components under different names: a system instruction, ordered user/assistant turns, a model identifier, and the generated text. The `Provider` interface lifts out that shared structure.

## The convergence on OpenAI-compatible APIs

In practice, a separate adapter per provider is not needed. Some time around 2024, "OpenAI-compatible" became the de facto interface for non-Anthropic LLM endpoints [18][19]. By 2026, Google Gemini, every local-model runner (Ollama, LM Studio, vLLM, llama.cpp), every aggregator (OpenRouter, Together, Replicate), and most cloud-hosted open-weights providers (Mistral, DeepSeek, Qwen, Kimi) all expose an HTTP endpoint that speaks OpenAI's request and response shape. Pointing the official `openai` Python SDK at their `base_url` with their `api_key` is enough, and everything works.

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

Anthropic also offers an OpenAI-compatible endpoint [21], but its own documentation says it is "primarily intended to test and compare model capabilities, and is not considered a long-term or production-ready solution." Routing Anthropic through the OpenAI shape drops every Anthropic-specific feature the agent relies on. The expensive loss is prompt caching: it stops working, so every request re-pays the full system-prompt cost that Chapter 4 worked to eliminate. Extended thinking still runs but its thinking deltas become unreadable, and the `usage` cache statistics that would confirm a hit always empty. Citations, structured outputs (`strict` is silently ignored), and PDF processing all degrade to plain text. That is why, for Anthropic specifically, the native SDK stays in `AnthropicProvider`.

So only two implementations are needed: a native Anthropic one, and a single OpenAI-compatible one that covers OpenAI itself, Gemini, OpenRouter, Ollama, LM Studio, and everything else.

## The `Provider` interface

The interface should hide which HTTP call is being made and expose only what `chat()` needs. Create a `providers/` directory in `agent/` and add `base.py`:

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

Some of the Python here is worth a closer look, especially coming from Java or C++.

`ABC` is the *abstract base class* marker from the `abc` module [22]. A class that inherits from `ABC` and contains at least one `@abstractmethod` cannot be instantiated directly and Python will raise `TypeError` on the attempt. This is Python's way of declaring an interface. Java would use the `interface` keyword and `abstract` methods; C++ uses pure virtual methods (`virtual void f() = 0;`) on a class with no implementation. 

Inside `stream`, the body is just `...` — the Ellipsis literal. The body is never executed for abstract methods, so a one-character placeholder is conventional. `pass` would also work but `...` is preferred in modern Python because it visually signals "intentionally empty."

Speaking about the interface design itself, the provider takes its configuration once, at construction time — model name, max tokens, API key, base URL — and exposes only `stream()`. That means caching policies, custom headers, timeout values, and SDK-specific kwargs all stay inside the provider. Had those things instead been passed into `stream()` on every call, the abstraction would leak in both directions: the agent would have to know about Anthropic's `cache_control` argument, and the provider would have no way to enforce its own defaults.

## Implementing `AnthropicProvider`

In the `providers/` folder, create `anthropic_provider.py`. The implementation is the existing `llm()` function wrapped into a class where `model` and `max_tokens` become construction-time parameters instead of hard-coded defaults.

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

The `cache_control={"type": "ephemeral"}` flag from Chapter 4 stays here because it is Anthropic-specific. OpenAI runs an automatic cache for prompts above a length threshold and requires no caller-side configuration [23], so the OpenAI provider will simply not pass anything cache-related. The `Provider` interface stays clean and each implementation handles caching the way its API expects.

## Implementing `OpenAIProvider`

Add the dependency:

```bash
uv add openai
```

This updates `pyproject.toml` and `uv.lock`. From now on `uv run main.py` will see the `openai` package without any additional steps.

Now create `openai_compatible_provider.py`. The provider has two new ideas: translating between the two message-shape conventions and iterating OpenAI's stream events instead of Anthropic's. Start with construction and translation, without streaming yet:

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
        if system: # <-- translation of how system prompt is passed
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens, # <-- different token budget parameter
            messages=oai_messages,
        )
        yield response.choices[0].message.content
```

Compared to the Anthropic version, the differences are small:

- The system prompt goes into the messages list as the first entry with `role="system"`, instead of being a separate parameter. That is the translation. 
- The token-budget parameter is `max_completion_tokens`, not `max_tokens`. OpenAI renamed it for newer models and Anthropic stayed with the original name. The provider abstracts the difference away so callers do not care.

Switching to real streaming requires changing only two lines:

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

Notice that OpenAI's streaming response is a plain iterator, not a context manager. Anthropic wraps the stream in `with` so it can clean up the underlying HTTP connection deterministically, while OpenAI's SDK relies on garbage collection. 

Also,  `chunk.choices[0].delta.content` can be empty. OpenAI's stream events are nested inside a list of choices, and a delta with no `content` is a signal that the chunk is something else, usually a tool-call delta or a finish-reason marker. The `if chunk.choices and chunk.choices[0].delta.content` guard is what filters down to actual text. The empty-content branches come back into play once tool calls are added.


## Wiring it up

In `main.py`, the `llm()` function is now obsolete because its body lives inside `AnthropicProvider.stream()`. Delete the function, delete the `import anthropic` and the module-level `client` it relied on, and replace them with a default provider:

```python
from providers.anthropic_provider import AnthropicProvider
from providers.base import Provider

DEFAULT_PROVIDER = AnthropicProvider()
```

`chat()` changes in only two places: it gains a `provider` parameter that defaults to `DEFAULT_PROVIDER`, and its streaming loop calls `provider.stream()` where the deleted `llm()` used to sit. Here is the function in full so the two edits are visible in context:

```python
def chat(provider: Provider | None = None) -> None:  # <-- new provider argument
    """Run an interactive chat loop, accumulating turns in a single messages list."""
    if provider is None:
        provider = DEFAULT_PROVIDER

    system = build_context()

    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break
        messages.append({"role": "user", "content": user_input})

        print("\nassistant: ", end="", flush=True)
        chunks: list[str] = []
        for text in provider.stream(messages, system=system):  # <-- was llm(messages, system=system)
            print(text, end="", flush=True)
            chunks.append(text)
        print("\n")

        messages.append({"role": "assistant", "content": "".join(chunks)})
```


## Verifying the swap actually works

Before building anything else on top of the `Provider` interface, run the same agent against a different backend.

Get a Gemini API key from Google AI Studio (free tier is enough for this) and add it to `.env`:

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

## Building automatic failover

The whole point of having a `Provider` interface is that a higher-level provider can wrap others. A `FallbackProvider` takes an ordered list of backends and tries them in turn. This way, if the first one raises a transient error, it moves on to the second.

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

The `yielded_anything` flag draws the line between a safe fallback and a corrupt one. Once any text has reached the user, switching providers would splice a second model's output onto the first mid-reply, so an error raised after the first delta aborts the whole call. Only pre-flight failures — auth errors, rate limits, connection errors that fire before any byte is streamed — fall back to the next provider.

The other thing to notice is what "Exception" means here. Catching everything is the simple version and is fine for a chapter. In production, retryable errors (HTTP 429 rate-limit, HTTP 5xx, connection timeouts) should be distinguished from non-retryable ones (HTTP 400 bad request, HTTP 401 auth) since the second class will fail on the next provider too, and trying it just adds latency to a guaranteed failure. Exercise 2 covers adding that distinction.

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

The agent now stops only when all three provider APIs are down at the same time.

One practical detail before testing: the snippet above assumes all three API keys are set. The `OpenAI()` and `anthropic.Anthropic()` constructors raise immediately if a key is missing, so a single missing variable will crash `main.py` at import time. With only one or two of the keys available, build the chain conditionally instead:

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

### Testing the failover

An actual outage is not needed to confirm the fallback works; simulating it by overriding an environment variable on a single command is enough. Each of the following invocations runs `main.py` with one or more API keys deliberately replaced by a wrong value, and which provider answered can be observed by asking the model who it is. The agent tries providers in the order they appear in the chain, so the first provider with a valid key wins.

1. Healthy run (baseline). All real keys, request goes to Anthropic because it is the first in the chain:

```bash
uv run main.py
# you: who made you?
# assistant: I'm Claude, made by Anthropic.
```

2. When Anthropic is down, OpenAI takes over.**

```bash
ANTHROPIC_API_KEY=bogus uv run main.py
# you: who made you?
# assistant: I'm ChatGPT, a large language model from OpenAI.
```

The first try (Anthropic) raises an authentication error immediately, before any byte has been streamed, so `FallbackProvider` walks to the next entry. The user sees no error, only a higher first-token latency.

3. Testing further, simulate the situation when Anthropic and OpenAI down so that Gemini takes over.**

```bash
ANTHROPIC_API_KEY=bogus OPENAI_API_KEY=bogus uv run main.py
# you: who made you?
# assistant: I'm Gemini, from Google.
```

4. Finally, all three down so the chain bottoms out.

```bash
ANTHROPIC_API_KEY=bogus OPENAI_API_KEY=bogus GEMINI_API_KEY=bogus uv run main.py
# you: who made you?
# RuntimeError: All providers failed; last error: AuthenticationError(...)
```

This is the only path that surfaces a user-facing error. Up to that point the abstraction has been silent.

What this exercise does not verify is the mid-stream guard (`yielded_anything`). Reproducing that requires either a real network drop in the middle of a reply or a more elaborate fault-injection rig. Exercise 2 walks through the production-grade error handling.

## A note on `litellm` and similar libraries

Anyone who has looked at the agent ecosystem has probably seen `litellm` [24]. This is a Python library whose entire job is to be the abstraction just written here, but for a hundred providers instead of two. This book choses not to use it for the following reasons.

First, the book's purpose is to understand how agents work. A library that hides every provider under one function call does not teach the foundational concepts. After this chapter, reading `litellm`'s source code and recognizing what it is doing becomes possible, and Exercise 4 walks through that mapping.

Second, and more important: in the real world, every provider library leaks. The leaks happen most often around tool calls, especially streaming tool calls (Chapter 7). The reason is structural: OpenAI streams a tool call by accumulating a JSON-string `arguments` field across many delta chunks — the model emits `{"loc` then `ation":"S` then `F"}` and the SDK reassembles. Anthropic streams the equivalent as a `tool_use` content block with typed `input_json_delta` events, then a `message_delta` with the final `stop_reason`. A library that wants to look like OpenAI to its callers has to either translate Anthropic's typed deltas into a fake JSON-string accumulator (losing the type information that made Anthropic's stream useful) or expose a leaky union type that callers have to unpack themselves. Both happen, and both are sources of bugs that production agents eventually hit and have to patch around.

Nanobot reached the same conclusion in production. It dropped `litellm` entirely in March 2026 (commit `3dfdab7` [25]) — a net −1,034 lines with all 593 tests still passing — because almost all of the removed code was workarounds for the leaks above. The proposal behind the change [26] is worth reading before reaching for any meta-library. Three reasons run through it. The dependency was heavier than the project it served: a ~30 MB library spanning 126 providers against the ~5 MB of the three SDKs nanobot actually used. Its translation layer turned provider errors into Pydantic warnings that matched nothing the API had returned, surfacing only after the request and response had already been rewritten, so there was nothing concrete to debug. And normalizing every provider to a lowest common denominator quietly dropped the provider-specific features — extended thinking, prompt caching, citations — that talking to each SDK directly keeps first-class.

The cost of going native is coverage: roughly 50 providers through native and OpenAI-compatible support, against litellm's 126. OpenRouter closes most of the gap, fronting hundreds of long-tail models behind one OpenAI-compatible endpoint that `OpenAIProvider` already speaks.

## Production reference

In nanobot, the production version of the `providers/` package built in this chapter is [`nanobot/nanobot/providers/`](https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers). The abstract `Provider` lives there as [`LLMProvider`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L91) in [`base.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py) — the same interface under a different name — with one file per backend behind it, exactly as `AnthropicProvider` and `OpenAIProvider` sit behind the interface here. The one structural addition is a [`registry.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/registry.py) and [`factory.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/factory.py) pair that builds providers from YAML configuration, where this chapter hardcodes a `DEFAULT_PROVIDER = ...` line.

As in the previous chapters, a few pieces are worth tracing once the chapter's own providers are written:

- **[`LLMProvider`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L91)** is the production version of the `Provider` abstract base class. Its streaming method is async, and the base class carries more than the chapter's `stream()`: the [`chat_stream_with_retry()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L527) wrapper every concrete provider inherits is where the chapter's bare `FallbackProvider` loop grows into a real retry-and-fallback policy.
- **[`_apply_cache_control()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L379)** in [`anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) is the production version of the single `cache_control={"type": "ephemeral"}` line. It walks the system, messages, and tools lists and places markers at each of the four allowed cache breakpoints, so a long tool catalog and a long conversation tail are cached independently.
- **[`chat_stream`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py#L1199)** in [`openai_compat_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py) is the production version of `OpenAIProvider`, the one file covering Gemini-via-compat, OpenRouter, Ollama, vLLM, LM Studio, DeepSeek, Mistral, and Kimi. Its shape is the one written here, plus handling for tool-call deltas (Chapter 7), cancellation (Chapter 9), and structured event emission (Chapter 22).

Nanobot uses `AsyncAnthropic` and `AsyncOpenAI`, with `max_retries=0` on the SDK and the centralized retry policy in `chat_stream_with_retry()` instead. The SDK's default exponential-backoff retry interacts badly with streaming: retrying a half-delivered stream produces duplicate output. Centralizing retries lets the policy stay aware of which streams are mid-flight.

Each `__anext__()` on the stream is wrapped in `asyncio.wait_for(..., timeout=idle_timeout_s)` (90 seconds by default, controlled by `NANOBOT_STREAM_IDLE_TIMEOUT_S`). Without that, a half-open TCP connection can hang the agent indefinitely because the SDK does not time out on its own.

## Exercises

1. **Native Gemini provider.** Install `google-genai` and write a `GeminiProvider` that uses Gemini's native SDK (the third example near the top of this chapter). It will be more fiddly than `OpenAIProvider` because the message shape is more different — `contents` instead of `messages`, `system_instruction` inside a config object, a different stream API. Once it works, weigh whether the native provider is preferable to using Gemini through `OpenAIProvider`. The right answer depends on whether any Gemini-specific feature is needed.

2. **Smarter `FallbackProvider`.** Extend the version from this chapter to distinguish retryable from non-retryable errors. Catch the SDK exceptions explicitly: `anthropic.RateLimitError`, `anthropic.APIStatusError`, `openai.RateLimitError`, `openai.APIStatusError`, `httpx.ConnectError`. Fall back on rate limits, 5xx status codes, and connection errors; re-raise everything else (auth errors, 4xx bad-request errors) immediately, since the next provider will fail the same way. Add exponential backoff between attempts. Test it by setting one provider's API key to a bogus value (auth failure → re-raise immediately, no fallback) and a different one to a real key (works on first try).

3. **Cost-routed provider.** Write a `RoutedProvider` that picks a cheaper backend for short messages and a stronger one for longer messages. Trivial heuristic: if `sum(len(m["content"]) for m in messages) < 2000`, use Haiku or `gpt-5-mini`; otherwise use Opus or `gpt-5`. This is a tiny example of the routing pattern Chapter 19 will revisit when subagents need to choose their own backend.

4. **Stretch: map `litellm` to the provider abstraction.** Clone `litellm` (the repository for this book already has it under [litellm/](../litellm/)) and open [`litellm/llms/base_llm/chat/transformation.py`](https://github.com/BerriAI/litellm/blob/487479eff76da099b358d06cfea0b6508fac4ef0/litellm/llms/base_llm/chat/transformation.py). Read `BaseConfig` carefully — focus on `transform_request`, `map_openai_params`, and `get_supported_openai_params`. Then open [`litellm/llms/anthropic/chat/transformation.py`](https://github.com/BerriAI/litellm/blob/487479eff76da099b358d06cfea0b6508fac4ef0/litellm/llms/anthropic/chat/transformation.py) and read `AnthropicConfig.transform_request`. Notice the direction of translation: in litellm, the *caller* always speaks OpenAI's shape, and each `transform_request` translates *outward* to the provider's native shape. Now refactor the `AnthropicProvider` to accept the same input convention — a flat OpenAI-style `messages` list where the system prompt is `messages[0]` with `role="system"`, instead of the current `(messages, system)` split. A small `_split_system(messages)` helper that pulls out the system message and reformats will be needed. Verify the existing `chat()` loop still works after the change. Then write a paragraph on which direction is easier to teach (caller-shape-as-OpenAI vs. caller-shape-as-`Provider`) and which is easier to scale to a hundred providers.

5. **Stretch: nanobot's prompt-caching strategy.** Open [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and find [`_apply_cache_control`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L379) (around line 379). Notice it places markers in three places: the tail of the system prompt, `messages[-2]`, and indexed tool entries. Implement a simplified version inside the `AnthropicProvider.stream` method: take `system` and `messages`, and attach `cache_control={"type": "ephemeral"}` to the system message *and* to `messages[-2]` (skip the second marker if there are fewer than three messages). Run a 6-turn conversation against a padded workspace (use the technique from Chapter 4 Exercise 5 to push the system prompt above the cache threshold), and print `final.usage.cache_read_input_tokens` after each turn. The numbers should climb starting at turn 3. Then predict why nanobot caches `messages[-2]` and not `messages[-1]`, validate the prediction by trying both, and compare the read counts. Hint: think about what changes between consecutive turns and what does not.

6. **Stretch: nanobot's error categorization.** Open [`nanobot/nanobot/providers/openai_compat_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py) and find where it categorizes errors. Notice how it distinguishes rate-limit errors from context-window-exceeded errors from generic transient errors. Add the same categorization to the `OpenAIProvider`: define a small enum (`ErrorCategory.RATELIMIT`, `ErrorCategory.CONTEXT_WINDOW`, `ErrorCategory.TRANSIENT`, `ErrorCategory.PERMANENT`) and a helper `_categorize(error: Exception) -> ErrorCategory`. Wire the categorization into the `FallbackProvider` from this chapter so that `CONTEXT_WINDOW` errors are re-raised immediately (the next provider has the same context limit), `RATELIMIT` and `TRANSIENT` trigger a fallback, and `PERMANENT` errors are re-raised. Test each branch.

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
