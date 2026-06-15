# Chapter 1: Your First LLM Call

This book starts by setting up the development environment and creating a first call to the LLM. This chapter also includes many references to future chapters to give a better sense of the book's scope, and it covers some general engineering concepts worth knowing before diving into the project.

## Setup

Before any code gets written, the required packages need to be installed. The uv package manager is the recommended choice [1]. Create a new folder for the agent, `cd` into it, and run

```bash
uv init
```

Running `ls -a` inside the folder shows what was created:

```
agent/
├── .git            - hidden folder for git management
├── .gitignore      - tells git what files should not be tracked in git history
├── main.py         - main entry point of the program
├── pyproject.toml  - project metadata: name, description, python version, dependency list etc.
├── .python-version
└── README.md

```

Next, add the required dependencies:

```bash
uv add anthropic
```

Behind the scenes, this creates a `.venv/` folder with the package and its transitive dependencies installed, and writes a `uv.lock` file alongside `pyproject.toml`. The two play different roles: `pyproject.toml` declares what the project _needs_, while `uv.lock` records exactly which versions uv actually resolved and installed down to the hash. `uv.lock` should be committed to git to guarantee that everyone engaged with a project gets an identical dependency tree. The file is plain TOML and human-readable, but it should be treated as generated output and left for uv to update [1].

The `anthropic` package [2] is the official Python SDK for Claude used throughout this chapter. SDK, or software development kit, is a set of software-building tools for a specific platform. The differences between SDKs and APIs are covered in this IBM blog [3].

Other providers have their own SDKs (`openai`, `google-genai`, etc.), explored in Chapter 5 by writing a thin abstraction to swap between providers in one line.

Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key. The best practice for storing the API key is to create an `.env` file in the root of the project and add environment variables like this:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Remember to add `.env` to `.gitignore` so that sensitive keys are not exposed when the project is pushed to GitHub. It is worth noting that even an API key pushed to a private repository can still be recovered by hackers, so this best practice is worth following in every project.

## The first call

Clean up `main.py` and add the following code that calls the LLM:

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello in five words."}]
)

print(response.content[0].text)
```

This imports the Anthropic SDK and initializes a client. The next call sends a message to `claude-opus-4-6` and gets the response back. Finally, the reply is printed. Run the script with

```bash
uv run main.py
```

The result is an error message like this:

```bash
TypeError: "Could not resolve authentication method. Expected either api_key or auth_token to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be explicitly omitted"
```

Although the API key was added to the `.env` file, the program has not yet been told to read it. That is the job of the `dotenv` library, installed with `uv` the same way as before:

```bash
uv add dotenv
```

Add two more imports at the top of the file and load environment variables with `dotenv.load_dotenv()`:

```python
# continue imports
import os
import dotenv

dotenv.load_dotenv()
# ...
```

Now the program can read the API key from the environment, and it can be passed to the Anthropic client explicitly:

```python
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

One such run returned `Hello there, how are you?`. The model received the message, generated a reply, and the SDK handed it back as a Python object.

## What is a message?

The body of the request — the part the model actually reads — is the `messages` parameter:

```python
messages=[{"role": "user", "content": "Say hello in five words."}]
```

A message is a dictionary with two fields: `role` and `content`. Three roles appear across this book.

- **`user`** — something the human (or, later, the world) sent in.
- **`assistant`** — something the model said.
- **`system`** — out-of-band instructions that shape who the model is and what it should do. Chapter 3 covers this.

A conversation is a list of messages in the order they happened and the model reads it every time. The API itself has no memory: each `messages.create` call is a stateless function from a list of messages to a reply. That sounds like a limitation, and in some ways it is, but it is also what makes the API easy to reason about. More detail is available here [4].

The `content` field, in this example, is just a string. It can also be a list of typed content blocks: text, images, tool calls, tool results.

## What is a token?

The model sees tokens instead of characters. A token is a chunk of text, usually a word, sometimes a word fragment or a single character. The string `"Say hello in five words."` is about seven tokens. The string `"antidisestablishmentarianism"` is also seven, despite being one English word. As a rough rule, one token is about four characters of English text — Anthropic's own estimate for Claude is closer to 3.5, and the exact number varies by language [5].

Tokens matter because:

1. **The model has a context window** — a maximum number of tokens it can see at once. For Claude Opus 4.7 it is one million, which is enormous, but it can still be exhausted by stuffing entire codebases into the prompt. The window covers everything: system prompt, full conversation history, current message, and the model's reply.

2. **Tokens cost money.** Input tokens (the ones sent) are cheaper than output tokens (the ones the model generates). Pricing changes all the time, so check the provider's site. The relevant point for now is that long conversations cost more than short ones, and the cost grows roughly linearly with conversation length on each turn. Chapter 15 revisits this with history compression.

3. **`max_tokens` is a cap on the reply.** A response that hits the cap will be cut off mid-sentence.

The SDK can count tokens ahead of time:

```python
client.messages.count_tokens(
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Say hello in five words."}],
).input_tokens
# 12
```

## What is a completion?

The thing `messages.create` returns is, confusingly, also called a message — a `Message` object representing the assistant's reply. (The Anthropic API uses "message" for both directions, while some providers call the response a "completion" or a "choice." This book uses the word response in prose to keep things readable.)

A response has three things worth knowing about:

```python
>>> response.content
[TextBlock(text='Hello! Welcome, how are you?', type='text')]
>>> response.stop_reason
'end_turn'
>>> response.usage
Usage(input_tokens=12, output_tokens=10, ...)
```

**`content`** is a list of content blocks. For a plain text reply, that list has exactly one block, which is a `TextBlock` object. When tools enter the picture (Chapter 7), the same field can hold `ToolUseBlock`s. Model's thinking will result in a `ThinkingBlock` object. So even though this chapter mostly reaches for `response.content[0].text`, that pattern should not be hardcoded into a loop yet. Chapter 7 writes it correctly the first time.

**`stop_reason`** explains why the model stopped generating. The most common values are:

- `end_turn` — the model finished a complete reply. 
- `max_tokens` — the model hit the cap mid-thought. The reply is truncated. Either raise `max_tokens` or stream the response.
- `tool_use` — the model wants to call a tool instead of replying.
- `refusal` — the model refused on safety grounds.

Chapter 9 treats stop reasons as first-class control-flow signals. For now, a quick check that the value is `end_turn` is enough.

**`usage`** reports how many input and output tokens the call consumed. Multiplying by the provider's per-token rate gives the cost. Chapter 24 assembles a dashboard to monitor usage.

## Production reference

Open [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and scroll to [`chat()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L540) near the bottom. It is a production version of what was just built: it builds a kwargs dict, calls `self._client.messages.create(**kwargs)`, and hands the result back. Everything else exists to feed those lines the right inputs and to make sense of the outputs.

Trace the following functions and map to what was implemented in the chapter:

- **[`_build_kwargs()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L416)** assembles the `model`, `messages`, `max_tokens`, `temperature`, and (optionally) `system`, `tools`, and `thinking` parameters from a higher-level request. This chapter hardcoded these.
- **[`_parse_response()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L485)** is the production version of `response.content[0].text`. Notice that it does not index `[0]` and rather walks the full block list, accumulates every text block, and routes `tool_use` and `thinking` blocks down separate paths. 
- **[`_handle_error()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L56)** classifies SDK exceptions into a structured response: status code, retry-after, error kind. 

Notice also that the client is `AsyncAnthropic`, not `Anthropic`. Since nanobot is the agent that talks to a chat channel while running a heartbeat while streaming a response, it cannot afford to block on a single network call. The synchronous SDK is fine for the `main.py` in this chapter. The retries are off at the SDK level (`max_retries=0` in the client constructor) and centralized one layer up. If both layers retried, a transient 429 would be retried twice over (once by the SDK, once by the wrapper) and back-off windows would multiply.

Inside [`nanobot/nanobot/providers/`](https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers), there is also one file per LLM backend ([`anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py), [`openai_compat_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py), [`azure_openai_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/azure_openai_provider.py), [`github_copilot_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/github_copilot_provider.py), etc.) and a [`base.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py) that defines the shared interface. This chapter's project arrives at this structure by the end of Chapter 5.

## Exercises

1. **Model identity.** Send the prompt `"Without searching the web, what model are you?"` to three different models — `claude-opus-4-6`, `claude-sonnet-4-6`, and `claude-haiku-4-5` — and compare the answers. Models are often unreliable narrators about themselves. This is a reminder to stay critical of the LLM as a source of truth.

2. **Inspect the response object.** The script reaches for `response.content[0].text` and ignores everything else. Take the same script and ask the model `"Think briefly before answering: what is 17 * 23?"`, but add `thinking={"type": "adaptive"}` as an extra argument to `messages.create`. Then `print(response.content)` raw without indexing and inspect the `type` field on each block. The list now contains more than one block, the first concrete hint of why hardcoding `[0]` was discouraged. While doing so, print `response.stop_reason` and `response.usage` and notice that the token cost has gone up: the model is being charged for thoughts that are now readable.

3. **Write a robust text extractor.** The trap flagged throughout — `response.content[0].text` — is one line away from being right. Write a helper `extract_text(response) -> str` that iterates `response.content`, keeps only blocks whose `type == "text"`, and joins their `text` fields. Run it against (a) a plain reply and (b) the adaptive-thinking response from Exercise 2. Then open [`_parse_response`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L485) in [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and compare. The shape will be very close.

4. **Count tokens before sending.** Use `client.messages.count_tokens(...)` to compute the input-token cost of a candidate prompt before sending it. Wrap this into a small helper `cost_estimate(messages, model)` that returns approximate dollars based on a hardcoded rate. Chapter 24 comes back to this.

5. **Prompt caching.** A long, repeated prefix — a system prompt, a knowledge document, a tool catalog — is paid for on every single call unless it is marked for caching. Open [`_apply_cache_control`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L379) in [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and observe how nanobot attaches `"cache_control": {"type": "ephemeral"}` to the last block of the system prompt. Reproduce that: pass a `system` parameter shaped like `[{"type": "text", "text": <a few hundred tokens of fake docs>, "cache_control": {"type": "ephemeral"}}]` and call the model twice with the same system prompt and a different user question each time. Print `response.usage` after each call. The first response will report `cache_creation_input_tokens`; the second should report `cache_read_input_tokens` instead, billed at roughly a tenth of the per-token rate. Prompt caching is one of the cheapest, highest-leverage levers in production, and Chapter 3 leans on it again when the system prompt starts growing.

## References

[1] _uv — an extremely fast Python package and project manager._ <https://docs.astral.sh/uv/>

[2] _Anthropic Python SDK reference._ <https://platform.claude.com/docs/en/api/sdks/python>

[3] _API vs. SDK: what's the difference?_ IBM Think. <https://www.ibm.com/think/topics/api-vs-sdk>

[4] _Using the Messages API — Multiple conversational turns._ Claude API documentation. <https://platform.claude.com/docs/en/build-with-claude/working-with-messages>

[5] _Glossary — Tokens._ Claude API documentation. <https://platform.claude.com/docs/en/about-claude/glossary>
