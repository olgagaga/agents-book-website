# Chapter 7: Tool Calling

Chapter 6's `loop.py` is a working agent  but the wire format is text: the model writes JSON inside its reply, the loop parses it, and streaming has to come off because there is no way to decide what kind of reply is coming until it ends.

By the end of this chapter, the loop streams the model's text live to the user as it arrives, and a single iteration can dispatch multiple tools at once with native tool API. 

## Anatomy of native tool calling

Anthropic and OpenAI both expose tool calling as a first-class feature of their respective chat endpoints [1][2]. The contract on both sides is the same: the available tools are declared in the request, and any tool calls the model decides to make come back in a dedicated slot of the response, separate from the natural-language text.

For Anthropic SDK, tool definitions go in a top-level `tools` parameter. Each tool has a name, a description, and a JSON Schema for its arguments:

```python
client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    system="You are concise.",
    tools=[{
        "name": "now",
        "description": "Return the current UTC time.",
        "input_schema": {"type": "object", "properties": {}},
    }],
    messages=[{"role": "user", "content": "what time is it?"}],
)
```

When the model wants to call a tool, the response's `content` is a list with one or more content blocks, each of type "text" or  "tool_use". The result sent back goes inside the next user message as a `{"type": "tool_result", "tool_use_id": ..., "content": ...}` block.

OpenAI employs a similar approach. Tools go under `tools` with a `"type": "function"` wrapper:

```python
client.chat.completions.create(
    model="gpt-5",
    tools=[{
        "type": "function",
        "function": {
            "name": "now",
            "description": "Return the current UTC time.",
            "parameters": {"type": "object", "properties": {}},
        },
    }],
    messages=[{"role": "user", "content": "what time is it?"}],
)
```

Tool calls come back on `message.tool_calls` as a list of objects with `id`, `function.name`, and `function.arguments` (a JSON string that has to be parsed). Results come back as a separate message with `role="tool"` and a matching `tool_call_id`.

Notice that the JSON Schema body is identical across providers: both use the standard `type` / `properties` / `required` vocabulary [3]. Both providers also give the model a separate channel for tool calls. 

## Defining tools

Create a dedicated `agent/tools/` folder. It will hold three things: a small `Tool` record type, a registry of available tools, and the implementations themselves. Start with `base.py`:

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]
```

`@dataclass` is the standard library's way to declare a typed record without writing `__init__`, `__repr__`, and `__eq__` by hand. A plain `dict` would be tempting but the typing is too loose: a typo like `tool["nmae"]` only surfaces at runtime, and an editor cannot autocomplete fields it does not know about. 

Each `Tool` carries four pieces: a name, a description that the model reads to decide whether to call it, an input schema (JSON Schema describing the arguments object), and a callable that takes the parsed arguments and returns a string. The string return is deliberate. Whatever the tool produces ultimately has to land inside a chat message, and chat messages are strings on the wire.

The tools from Chapter 6's main text and Exercise 1 can now be defined in `tools.py`:

```python
from datetime import datetime, timezone

def _now(args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


def _wordcount(args: dict) -> str:
    return str(len(args["text"].split()))

```
Finally, `registry.py` will be used to define the set of available tools for the model:


```python
from tools.tools import _now, _wordcount
from tools.base import Tool


TOOLS: list[Tool] = [
    Tool(
        name="now",
        description="Return the current UTC time as an ISO 8601 string.",
        schema={"type": "object", "properties": {}, "required": []},
        run=_now,
    ),
    Tool(
        name="wordcount",
        description="Count whitespace-separated words in the given text.",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        run=_wordcount,
    ),
]

```

A small `find_tool` lookup helper rounds out the registry. When the loop sees `reply.tool_calls = [ToolCall(name="wordcount", ...)]` and needs to actually run something, it has to turn that name back into the `Tool` object whose `run` callable to invoke. The registry is small enough that a linear scan is fine for now:

```python
def find_tool(name: str) -> Tool | None:
    return next((t for t in TOOLS if t.name == name), None)
```

With this setup, `loop.py` can be cleaned up: drop `TOOL_INSTRUCTIONS`, drop the in-file `TOOLS` registry, drop `_parse_tool_call` and `_build_agent_system`. The provider will declare tools to the model directly, on the same wire format the model was trained to fill [1][2].

## A new shape for the Provider

Chapter 5's `Provider.stream` returns an `Iterator[str]` of text deltas. That no longer fits: a reply can now carry tool calls alongside its text, so the provider needs a return type that holds both. Two small dataclasses in `agent/providers/base.py` cover it:

```python
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
```

`Reply.text` is the full assistant text concatenated; `Reply.tool_calls` is the list of structured calls the model wants executed, empty when the model is done.

Returning a single `Reply` forces one more change: streaming has to move out of the return type. Chapter 5 returned an `Iterator[str]` and let the caller pull text chunks out, but a `Reply` can only be handed back once the model has finished — that is the only moment its `tool_calls` list fully exists. To keep text flowing live in the meantime, `call` takes an `on_text_delta` callback: as each delta arrives off the wire, `on_text_delta(delta)` fires and the caller prints it immediately. A caller that only wants the final structured reply passes `None`, and the callback is skipped.

The messages flowing into `call` stay in the flat OpenAI shape from Chapter 5 — a list of `{"role": ..., "content": ...}` dicts — which keeps the loop provider-agnostic and pushes any format translation into the providers themselves. With the return type and the streaming callback settled, the abstract method is short:

```python
from typing import Callable

from tools import Tool

class Provider(ABC):
    """A streaming LLM backend with tool-call support."""

    @abstractmethod
    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        """Run one model turn and return the final reply.

        Streams text deltas through ``on_text_delta`` as they arrive.
        ``Reply.text`` carries the same text concatenated.
        """
        ...
```


## AnthropicProvider with tools

Anthropic's API does not accept the shape we defined earlier. Instead, assistant messages must be a list of content blocks interleaving `text` and `tool_use`, and tool results come back inside a user message as a `tool_result` block. 

Therefore, provider must first serve as a translation layer and then submit the translated request and reassemble  a single `Reply` from Anthropic's event-based response stream.

### Translation

The `messages` list that arrives at `provider.call` is the running conversation the loop has built up over previous iterations. The job of `_to_anthropic_messages` is to walk that list and rewrite each entry in the shape Anthropic expects, dispatching on `m["role"]`:

```python
class AnthropicProvider(Provider):
    # --- __init__ method above ---
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                pass
            elif m["role"] == "assistant":
                pass
            elif m["role"] == "tool":
                pass
```

User messages translate verbatim — Anthropic accepts the same `{"role": "user", "content": "..."}` shape the internal format already uses:

```python
class AnthropicProvider(Provider):
    # --- __init__ method above ---
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                pass
            elif m["role"] == "tool":
                pass
```

Assistant messages are where the shapes diverge. In the internal format the model's text reply lives in `content` and any tool calls the model wanted live in a separate `tool_calls` list. Anthropic expects all of that as a single `content` list of typed blocks, with text and `tool_use` interleaved. The blocks accumulate in order — text first if present, then one `tool_use` block per call — and the assistant message is emitted with the assembled list. Field names also differ on a single piece: the internal `args` becomes Anthropic's `input`.

```python
class AnthropicProvider(Provider):
    # --- __init__ method above ---
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                pass
```

Finally, `role="tool"` is a wire-format invention of OpenAI that Anthropic does not share. In Anthropic's world, tool results ride back as a `tool_result` content block inside the next user message. The translation flips the role from `tool` to `user`, wraps the observation in a `tool_result` block, and threads the matching `tool_use_id` through so Anthropic can pair the result with the original call:

```python
class AnthropicProvider(Provider):
    # --- __init__ method above ---
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }]})
        return out
```

Tool definitions need translation too, for a simpler reason: the `Tool` dataclass is provider-neutral, with a generic `schema` field that each provider's API wants to receive under a different name. Anthropic uses `input_schema`, sitting at the top level of the tool entry:

```python
from tools.base import Tool

class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    @staticmethod
    def _to_anthropic_tools(tools: list[Tool]) -> list[dict]:
        return [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema,
        } for t in tools]
```

### Streaming

With translation done, the actual `call` method comes next. The rename from `stream` to `call` is intentional: in Chapter 5 the method returned an iterator of text deltas, so `stream` was an accurate name. Here the method returns a structured `Reply` at the end and merely delivers text deltas through `on_text_delta` along the way.

Anthropic's streaming API is event-based [1]. Instead of yielding plain text fragments, the SDK iterator yields typed events describing what is happening on the wire: a new content block starting, a delta inside the current block, a block ending, the message finishing, usage data arriving, and so on. Of all of these, this chapter cares about exactly two:

- `content_block_start` — fires once at the beginning of each new content block. For `tool_use` blocks, this is where the tool's `id` and `name` arrive (the arguments come later, as deltas). Text blocks need no setup at this event.
- `content_block_delta` — fires for the streamed contents of an in-progress block. Two flavors of delta are relevant here: `text_delta` (a chunk of assistant text) and `input_json_delta` (a fragment of a tool's argument JSON).

The other event types (`message_start`, `message_delta`, `content_block_stop`, `message_stop`, usage events) still flow past in the same loop and are silently ignored.

The first step renames the function and adds the two new arguments — `tools` (the list of available tools) and `on_text_delta` (the streaming callback introduced on the abstract `Provider`) — and changes the return type from `Iterator[str]` to `Reply`:

```python
from typing import Callable
from providers.base import Provider, Reply

class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
```

It begins by accumulating the keyword arguments for the call to Anthropic's client. The model and `max_tokens` come from the `__init__`, and the messages are translated through `_to_anthropic_messages`:


```python
class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }
```

Then, if a system prompt is set, it is passed through with a `cache_control` marker so Anthropic caches the prefix between turns — the same prompt-caching technique Chapter 5 introduced. Tools, if any, are translated through `_to_anthropic_tools`:

```python
class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }

        if system:
            kwargs["system"] = system
            kwargs["cache_control"] = {"type": "ephemeral"}
        
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(list(tools))
```

The next step is to accumulate state across the stream as it arrives, in three structures:

- `text_parts: list[str]` — every `text_delta`'s text, appended in order. Joined at the end to form `Reply.text`.
- `tc_by_index: dict[int, ToolCall]` — one `ToolCall` per `tool_use` block, indexed by Anthropic's content-block index. It is seeded with `id` and `name` at `content_block_start` and filled in with `args` at the end.
- `partial_args: dict[int, str]` — the in-progress argument JSON string for each tool call. Anthropic does not send tool arguments as a finished JSON object; it streams them as a sequence of byte-level fragments (`{"l`, then `oc`, then `ation":"S`, then `F"}`) so that the model's tool-argument generation is just as streamable as its text generation. `json.loads` cannot run until the block is complete, so the fragments are concatenated here and parsed once at the end.

The stream itself opens with the SDK's `messages.stream` context manager, which yields events one by one:

```python
from providers.base import Provider, Reply, ToolCall

class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }

        if system:
            kwargs["system"] = system
            kwargs["cache_control"] = {"type": "ephemeral"}

        if tools:
            kwargs["tools"] = self._to_anthropic_tools(list(tools))

        text_parts: list[str] = []
        tc_by_index: dict[int, ToolCall] = {}
        partial_args: dict[int, str] = {}

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                pass

```

On `content_block_start` the only action happens when the new block is a `tool_use` — that is the one event where Anthropic ships the tool's `id` and `name`. The handler seeds `tc_by_index` at the block's index with a `ToolCall` carrying that id and name, and prepares an empty string in `partial_args` for the upcoming JSON fragments:

```python
from providers.base import Provider, Reply, ToolCall

class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        # --- kwargs collection ---
        # --- arrays initialization ---

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tc_by_index[event.index] = ToolCall(
                            id=block.id, name=block.name, args={},
                        )
                        partial_args[event.index] = ""
                elif event.type == "content_block_delta":
                    pass

```

On `content_block_delta` the two delta types that matter are `text_delta` and `input_json_delta`. Text deltas get appended to `text_parts` and forwarded to `on_text_delta` immediately, so the caller can stream them to the terminal as they arrive. JSON deltas get concatenated into the right `partial_args` slot, addressed by the same `event.index` that identified the matching `tool_use` block at start-time:


```python
from providers.base import Provider, Reply, ToolCall

class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        # --- kwargs collection ---
        # --- arrays initialization ---

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tc_by_index[event.index] = ToolCall(
                            id=block.id, name=block.name, args={},
                        )
                        partial_args[event.index] = ""
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_parts.append(delta.text)
                        if on_text_delta:
                            on_text_delta(delta.text)
                    elif delta.type == "input_json_delta":
                        partial_args[event.index] += delta.partial_json

```

After the stream finishes, every `partial_args` slot holds the complete JSON string for one tool call. Each one is parsed and assigned to the matching `ToolCall.args`, defaulting to `{}` when the model produced no fragments at all. The final `Reply` is then assembled from the joined text and the list of accumulated tool calls:

```python
import json

class AnthropicProvider(Provider):
    # --- __init__ method above ---
    # --- _to_anthropic_messages method above ---
    # --- _to_anthropic_tools method above --- 

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        # --- kwargs collection ---
        # --- arrays initialization ---

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                # --- events processing
        
        for i, raw in partial_args.items():
            tc_by_index[i].args = json.loads(raw) if raw else {}

        return Reply(text="".join(text_parts), tool_calls=list(tc_by_index.values()))
```

The streaming the loop lost in Chapter 6 is restored: every `text_delta` event is forwarded through `on_text_delta` the moment it arrives, so a caller printing each delta sees the model's output live.

## OpenAIProvider with tools

The internal message format already is OpenAI shape — `role` plus `content` plus optional `tool_calls` — so there is no message-level translation work to do here. The only translation is on tools (rename `schema` to `parameters` and wrap the entry in a `{"type": "function", ...}` envelope), and on the streamed tool-call arguments, which arrive as JSON fragments rather than as a finished object — the same fragment-streaming behaviour as Anthropic's, for the same reason.

First, the tool translation:

```python
from tools.base import Tool

class OpenAIProvider(Provider):
    # --- __init__ method above ---

    @staticmethod
    def _to_openai_tools(tools: list[Tool]) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        } for t in tools]
```

The same `stream` → `call` rename happens here, with the same expanded signature (adding `tools` and `on_text_delta`, returning `Reply`). It begins by assembling the request keyword arguments. One difference from Anthropic is worth noting: OpenAI delivers the system prompt as the first entry in the `messages` list with `role="system"`, rather than as a separate `system` argument:

```python
from typing import Callable
from providers.base import Provider, Reply

class OpenAIProvider(Provider):
    # --- __init__ method above ---
    # --- _to_openai_tools method above ---

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        kwargs: dict = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": oai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(list(tools))
```

The streaming loop has the same two responsibilities as the Anthropic version: pass each chunk of text through `on_text_delta` the moment it arrives — and remember it in `text_parts` for the final `Reply.text` — and accumulate tool-call argument JSON fragments per call until the stream ends. The shape of the stream is different from Anthropic's. OpenAI emits flat `chat.completion.chunk` objects, each with a `delta` carrying whatever changed in this chunk: text content, a new tool-call id, a tool-call name, or more argument bytes. Each piece slots into a `partial` dict keyed by `tc_delta.index`, and the `ToolCall` objects are assembled at the end when every fragment has arrived.


```python
import json
from providers.base import Provider, Reply, ToolCall

class OpenAIProvider(Provider):
    # --- __init__ method above ---
    # --- _to_openai_tools method above ---

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:

        # --- handle system prompt and collect kwargs ---

        text_parts: list[str] = []
        partial: dict[int, dict] = {}

        for chunk in self.client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                if on_text_delta:
                    on_text_delta(delta.content)
            for tc_delta in (delta.tool_calls or []):
                slot = partial.setdefault(tc_delta.index, {"id": "", "name": "", "args": ""})
                if tc_delta.id:
                    slot["id"] = tc_delta.id
                fn = tc_delta.function
                if fn and fn.name:
                    slot["name"] = fn.name
                if fn and fn.arguments:
                    slot["args"] += fn.arguments

        tool_calls = [
            ToolCall(id=p["id"], name=p["name"], args=json.loads(p["args"] or "{}"))
            for p in partial.values()
        ]
        return Reply(text="".join(text_parts), tool_calls=tool_calls)
```

One last thing to handle. OpenAI's streaming API can emit chunks whose `choices` array is empty — typically a final usage-summary chunk that carries prompt and completion token counts for the whole call rather than another piece of message content, sent when `stream_options={"include_usage": True}` is set (some compatibility gateways inject one whether asked for it or not) [2]. The loop above reads `chunk.choices[0]` unconditionally, which would crash on that chunk. The guard is a one-liner at the top of the loop body: if `choices` is empty, skip the chunk and move on.


```python
class OpenAIProvider(Provider):
    # --- __init__ method above ---
    # --- _to_openai_tools method above ---

    def call(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[Tool] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Reply:

        # --- handle system prompt and collect kwargs ---

        text_parts: list[str] = []
        partial: dict[int, dict] = {}

        for chunk in self.client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            # --- rest of the loop code ---

        # --- tool call and reply assembly and return ---
```

`FallbackProvider` from Chapter 5 also needs to be ported to the new shape. The change is small — match the new `call` signature, return the inner provider's `Reply` directly, and keep the mid-stream guard so the agent does not fall back to a second provider after the first has already streamed text to the user. That port is Exercise 6.

## The loop, simplified

Finally, `agent/loop.py` gets finalized to handle native tool calls. The first change modifies `agent_step`'s signature to accept the `on_text_delta` argument:

```python
from typing import Callable

def agent_step(
    user_message: str,
    history: list[dict],
    provider: Provider,
    system: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:

    messages = list(history)
    messages.append({"role": "user", "content": user_message})
```

Since streaming now happens inside the provider and a single assembled `Reply` comes back, the inner-loop branch changes shape. When the model returns no tool calls, the reply is the final answer and its text is returned — taking the role of Chapter 6's `if call is None` branch:

```python
from tools.registry import TOOLS
from providers.base import Provider, Reply

def agent_step(
    user_message: str,
    history: list[dict],
    provider: Provider,
    system: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:

    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_ITERATIONS):
        reply: Reply = provider.call(
            messages, system=system, tools=TOOLS, on_text_delta=on_text_delta,
        )

        if not reply.tool_calls:
            return reply.text

```

The tool calls are handled in two steps: append the assistant message (with its `tool_calls`) to the conversation, then run each requested tool and append its result:


```python
from tools.registry import TOOLS, find_tool

def agent_step(
    user_message: str,
    history: list[dict],
    provider: Provider,
    system: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_ITERATIONS):
        # --- collect reply ---

        messages.append({
            "role": "assistant",
            "content": reply.text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in reply.tool_calls
            ],
        })
        for tc in reply.tool_calls:
            tool = find_tool(tc.name)
            if tool is None:
                observation = f"Error: no tool named {tc.name!r}."
            else:
                try:
                    observation = tool.run(tc.args)
                except Exception as exc:
                    observation = f"Error: {type(exc).__name__}: {exc}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": observation,
            })

    return "I exceeded the maximum number of steps without producing a final answer."
```

Compared to Chapter 6, the model's intent now lives in `reply.tool_calls` directly, with no parsing on the loop's side. The for-loop over `reply.tool_calls` runs each tool the model requested and appends matching `role="tool"` messages back into the conversation; the provider takes care of translating those back to its native wire format inside `call`.

With `_build_agent_system()` removed — the model no longer needs to be told how to call tools in prose, only which tools exist — `chat()` in `main.py` changes to call the plain `build_context()` from Chapter 3 directly:

```python
from loop import agent_step
from context import build_context

def chat(provider: Provider | None = None) -> None:
    # --- code above building a system prompt
    system = build_context()
    # --- code below ---
```

## Wiring `chat()` with streaming

Chapter 6 disabled streaming in `chat()` because the loop could not tell whether a reply was a tool call or a final answer until the whole reply had arrived — and a half-streamed JSON tool call printed live to the terminal would have been worse than no streaming at all. Native tool calls fix that at the wire-format layer: text and tool-use blocks are different events from the first byte, so text can stream the moment it arrives with no risk of accidentally streaming a JSON tool call by mistake:

```python
def chat(provider: Provider | None = None) -> None:
    """Run an interactive chat loop, streaming the agent's text deltas as they arrive."""
    # --- code above ----
    while True:
        # --- handle user input ---

        print("\nassistant: ", end="", flush=True)
        reply = agent_step(
            user_input, messages, provider, system,
            on_text_delta=lambda delta: print(delta, end="", flush=True),
        )
        print("\n")

        # --- append messages to a message history ---
```

`on_text_delta=lambda delta: print(...)` is the sink that turns each delta into a printed character — the concrete caller for the callback declared back on the abstract `Provider`. Anything the model says, preamble or final answer, now streams to the terminal in real time.

The long-term `messages` history still keeps only the user prompt and the final assistant text. The intermediate scratchpad inside `agent_step` — every tool-call message, every observation — is discarded when `agent_step` returns.

## A worked example

Run `uv run main.py`:

```
chat — Ctrl-D or empty line to exit

you: what is the current UTC time, to the second?

assistant: The current UTC time is **2026-05-11T16:23:53** (May 11, 2026, 4:23:53 PM UTC).
you: ^D
```

What happened under the hood: in iteration 1, the model emitted nothing but a `tool_use` block — no preamble text, hence no `on_text_delta` calls, hence nothing on screen — and the `now` tool ran silently in the dispatch loop. In iteration 2, with the tool result threaded back into the conversation, the model produced the final answer text; each `text_delta` event flowed through `on_text_delta` to `print(..., flush=True)`, which is why the answer appeared character by character rather than as one blob at the end of the call. The blank lines around `assistant:` are just `print` formatting from `chat()`; everything in between was streamed live.

Now ask something requiring two tool calls in one turn:

```
you: what time is it, and how many words are in "the quick brown fox jumps over the lazy dog"?

assistant: 

Let me get both of those for you at once.The current UTC time is **2026-05-11 16:26:13**, and the sentence "the quick brown fox jumps over the lazy dog" contains **9 words**.

you: ^D
```

A capable model emits both tool calls in a single response. The loop's `for tc in reply.tool_calls` runs both, appends two `role="tool"` messages, and the next iteration sees both results at once. Running the same query against `OpenAIProvider(model="gpt-5")` confirms the abstraction holds: same output, same number of iterations.

To make the trace visible during development, add a `print` inside `agent_step` after each `reply` is received:

```python
print(f"\n[iter] text={reply.text[:60]!r} tools={[(tc.name, tc.args) for tc in reply.tool_calls]}")
```

The two-tool query then prints something like:

```
chat — Ctrl-D or empty line to exit

you: what time is it, and how many words are in "the quick brown fox jumps over the lazy dog"?

assistant: 

Let me check both of those for you at once.
[iter] text='\n\nLet me check both of those for you at once.' tools=[('now', {}), ('wordcount', {'text': 'the quick brown fox jumps over the lazy dog'})]
The current UTC time is **2026-05-11 16:27:53**, and the sentence "the quick brown fox jumps over the lazy dog" contains **9 words**.
[iter] text='The current UTC time is **2026-05-11 16:27:53**, and the sen' tools=[]


you: ^D
```

The trace makes the payoff concrete. The model emitted both tool calls in iteration 1 as two structured `tool_use` blocks, and the API delivered them already parsed. The loop got back a `Reply` whose `tool_calls` list held two fully-populated `ToolCall` objects, with ids, names, and JSON arguments already parsed into Python dicts. The single `for tc in reply.tool_calls` line dispatched both in one iteration. The bookkeeping the Chapter 6 design did by hand is here two attribute accesses on a dataclass.

Now try a *two-turn* exchange where the second turn depends on the first:

```
chat — Ctrl-D or empty line to exit

you: what is the current UTC time?

assistant: The current UTC time is **2026-05-11T16:31:02Z**.

you: how long has it been since I asked?

assistant: I don't have a record of when you asked the previous question — let me check the time now and you can compare it yourself: it is currently **2026-05-11T16:31:38Z**.

you: ^D
```

The second reply is visibly weaker than the first. Sometimes the model invents a plausible elapsed time ("about thirty seconds"), sometimes it issues a fresh `now` and apologises for not being able to compare, sometimes it asks the user to repeat the original question. None of these are wrong, exactly — the model is doing the best it can with the information available — but compared to the single-turn traces above, something is clearly missing.

The cause is the `agent_step` shape just built. The first line of the function copies `history` into a local `messages` list; the loop appends freely to that copy — user prompt, assistant tool-call messages, synthetic tool-result observations — but only `reply.text` ever leaves the function. When `agent_step` returns, the scratchpad is garbage-collected. The caller in `chat()` records only the final user prompt and the final assistant text, so on turn two the model sees `"what is the current UTC time?" → "The current UTC time is …"` followed by the new question — with no trace of the `now` tool call that produced the timestamp. From the model's point of view, the previous turn could have come from anywhere. Chapter 8 fixes this by mutating `messages` in place from inside `agent_step` so the scratchpad survives across turn boundaries, and it pairs that change with a size budget on observations so the longer history does not run away with the context window.


## Production reference

Open [`nanobot/nanobot/providers/`](https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers). Three pieces of it map directly to what this chapter built.

- **[`base.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py)** — the production analogues of the toy's `ToolCall` and `Reply` live here: [`ToolCallRequest`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L19) and [`LLMResponse`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py#L48). `LLMResponse` carries the same `content` and `tool_calls`, plus `finish_reason` (the stop reason Chapter 9 will care about), `usage` (token counts), `reasoning_content` (Kimi/DeepSeek-style chain-of-thought), and `thinking_blocks` (Anthropic extended thinking). The shape is exactly what the chapter is building toward, with later-chapter features pre-wired in.
- **[`anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py)** — [`_convert_messages`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L121) is the production version of `_to_anthropic_messages`. The core idea is identical: walk the OpenAI-shape list, repack assistant messages into content blocks, repack `role="tool"` into user-side `tool_result` blocks. The supporting helpers ([`_assistant_blocks`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L179), [`_tool_result_block`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L164)) factor out the per-role work the same way the chapter did inline.
- **[`openai_compat_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py)** — the OpenAI-shape streaming loop, recognisable from the toy: a `partial` dict keyed by tool-call index, per-chunk text-delta forwarding, the same empty-`choices` guard just added, plus several years of edge-case handling on top.

A few production nuances are visible in those files but invisible from the toy:

**Defensive message normalization.** `_convert_messages` also handles edge cases the toy ignores: image blocks (`_convert_image_block`), merging consecutive same-role messages while preserving `tool_use` boundaries (`_merge_consecutive` plus `_has_tool_use` — Anthropic rejects an unmatched `tool_use` / `tool_result` pair), and enforcing role alternation when injection messages get spliced mid-turn (`_enforce_role_alternation`, in `base.py`). Each guard traces back to a real production failure.

**Cross-provider id sanity.** [`openai_compat_provider.py::_normalize_tool_call_id`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py#L367) hashes long tool-call ids down to a fixed 9-character form. OpenRouter, Azure OpenAI, and various gateway proxies sometimes lengthen or reformat tool-call ids in transit, and a few stricter providers reject ids that exceed an internal limit. Threading the original id through verbatim — which the toy does — is fine for direct Anthropic and OpenAI traffic, and breaks subtly when traffic is fronted by an aggregator.

**Tool dispatch concurrency.** When a single iteration produces several tool calls — the two-tool query above is a small example — nanobot dispatches them concurrently with `asyncio.gather` inside `AgentRunner._execute_tools`. The toy's sync `for tc in reply.tool_calls` loop runs them one after the other, which is fine while every tool is in-process and returns instantly. The moment a tool reaches out to a slow HTTP API, that serialization becomes a visible stall — Chapter 10 introduces `asyncio` and revisits this dispatch.

## Exercises

1. **Malformed tool-call JSON.** The `json.loads(raw)` call at the end of `AnthropicProvider.call` will crash the whole agent step if the model produces a tool call with broken JSON arguments. While rare on production models, it is common when experimenting with smaller local ones. Catch the failure: when `json.loads` raises, set the offending `ToolCall.args` to `{}` and append a synthetic error string to `text_parts` so the loop's next iteration can react ("the model produced invalid JSON for tool `<name>`; it should retry"). Verify the fix against `OpenAIProvider` too since it has the same parse step.

2. **A third tool.** Add `random_int` to `TOOLS`: takes `{"low": int, "high": int}` and returns a random integer in `[low, high]` inclusive. Encode the `low ≤ high` invariant in the schema using `minimum`/`maximum` constraints (look up JSON Schema's [3] support for these). Ask the model to roll three values between 1 and 100 in one turn and watch the loop run all three calls back-to-back inside a single iteration. (Real parallelism — `asyncio.gather`-ing the calls — is what nanobot's `_execute_tools` does. The sync loop runs them one after the other, which is fine for in-process tools and becomes Chapter 10's problem once tools start blocking.)

3. **Tool-result errors.** In the current loop, a tool that raises returns `Error: <type>: <message>` as an observation. The model usually recovers (it sees the error, apologizes, tries something else), but sometimes it loops forever on the same broken call. Modify the loop so that if the same tool name fails twice in a row with the same arguments, the loop terminates with a user-visible error instead of trying a third time. Be deliberate about what counts as "the same call."

4. **Tool choice.** Both Anthropic and OpenAI accept a `tool_choice` parameter that forces or forbids tool use. Add an optional `tool_choice` argument to `Provider.call` and thread it through both providers' translations (`{"type": "any"}` / `{"type": "auto"}` / `{"type": "tool", "name": "now"}` for Anthropic; the equivalent OpenAI shapes are documented at [2]). Use it to write a one-shot helper that asks the model "what time is it?" and forces a `now` call, then returns the result string directly.

5. **Streaming a thinking spinner.** `on_text_delta` lets the user see text as it streams, but tool calls are silent — the user sees no movement while the loop is between text bursts. Add a tool-call lifecycle hook to `Provider.call` (`on_tool_call_start: Callable[[str], None]`) that fires when a tool-use block first appears in the stream. Wire it from `chat()` to print a `[calling now()]…` spinner between text bursts. Verify it works for Anthropic and for OpenAI.

6. **Stretch: rewrite `FallbackProvider` for the new shape.** Update `FallbackProvider` so that `call` matches the new abstract method. The mid-stream guard from Chapter 5 still applies — after the first `on_text_delta` invocation, an exception must terminate the call rather than fall back. Use a wrapper closure around the caller's `on_text_delta` to flip a flag the outer `try/except` can read. Test by running with a bogus `ANTHROPIC_API_KEY` (auth fails before any byte) and confirming `OpenAIProvider` takes over silently.

7. **Stretch: read `_convert_messages`.** Open [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and read [`_convert_messages`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L121) plus [`_assistant_blocks`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L179) and [`_tool_result_block`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L164). Find at least three things they do that the chapter's `_to_anthropic_messages` does not. For each, write one sentence explaining what real-world failure mode it guards against — the comments and surrounding helpers carry most of the story. (Hint: look for `_has_tool_use`, `_merge_consecutive`, and the image-block conversion.)

## References

[1] *Tool use with Claude.* Claude API documentation. <https://platform.claude.com/docs/en/build-with-claude/tool-use>

[2] *Function calling.* OpenAI documentation. <https://platform.openai.com/docs/guides/function-calling>

[3] *JSON Schema.* <https://json-schema.org/>
