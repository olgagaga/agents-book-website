# Chapter 1: Your First LLM Call

We start this book by setting up the development environment and creating our first call to the LLM. You will also see a lot of references to the future chapters to get a better sense of the scope of this book. This chapter also covers some general engineering concepts worth knowing before diving into the project. 

## Setup

Before we start writing code, let's install the required packages. I'd recommend the uv package manager [1]. Create a new folder for your agent, `cd` into it, and run

```bash
uv init
```

Running `ls -a` inside the folder lets us see what was created:

```
agent/
├── .git            - hidden folder for git management
├── .gitignore      - tells git what files should not be tracked in git history
├── main.py         - main entry point of the program
├── pyproject.toml  - project metadata: name, description, python version, dependency list etc.
├── .python-version
└── README.md   

```

Let's now add the required dependencies:

```bash
uv add anthropic
```

Behind the scenes, this creates a `.venv/` folder with the package and its transitive dependencies installed, and writes a `uv.lock` file alongside `pyproject.toml`. The two play different roles: `pyproject.toml` declares what your project *needs* (for example, "some compatible version of `anthropic`"), while `uv.lock` records exactly which versions uv actually resolved and installed down to the hash. Commit `uv.lock` to git: that is what guarantees a teammate (or your future self on a new machine) gets the same dependency tree you did. The file is plain TOML and human-readable, but treat it as generated output and let uv update it [1].

The `anthropic` package [2] is the official Python SDK for Claude that we will use in this chapter. SDK, or software development kit, is a set of software-building tools for a specific platform. You can dive into the differences between SDKs and APIs in this IBM blog [3].

Other providers have their own SDKs (`openai`, `google-genai`, etc.) which we will explore in Chapter 5 by writing a thin abstraction to swap between the providers in one line.

Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key. The best practice for storing the API key is to create an `.env` file in the root of your project and add environment variables like this:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Do not forget to add `.env` to your `.gitignore` so that your sensitive keys are not exposed when the project is pushed to GitHub. It is worth noting that even if you pushed your API key to a private repository, there are still ways to recover them by hackers. So make sure to follow this best practice in every project you create.

If you would rather use a different provider — OpenAI, OpenRouter, a local model via Ollama — you can. The shape of what we write in this chapter is the same everywhere; only the SDK and model name change. Chapter 5 generalizes properly.

## The first call

Let's make our first call to the LLM. Clean up `main.py` and put the following code here:

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

We import the Anthropic SDK and initialize a client. The next call sends a message to `claude-opus-4-6` and gets the response back. Finally, we print it. Run your script with

```bash
uv run main.py
```

You should see an error message like this:

```bash
TypeError: "Could not resolve authentication method. Expected either api_key or auth_token to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be explicitly omitted"
```

Although we added the API key to the `.env` file, we have not yet told our program to read it. For that, we will use the `dotenv` library, installed with `uv` the same way as before:

```bash
uv add dotenv
```

Add two more imports at the top of your file and load environment variables with `dotenv.load_dotenv()`:

```python
# continue imports
import os
import dotenv

dotenv.load_dotenv()
# ...
```

Now our program can read the API key from the environment, and we can pass it to the Anthropic client explicitly:

```python
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

In my case, I got `Hello there, how are you?`. The model received our message, generated a reply, and the SDK handed it back to us as a Python object.

## What is a message?

The body of the request — the part the model actually reads — is the `messages` parameter:

```python
messages=[{"role": "user", "content": "Say hello in five words."}]
```

A message is a dictionary with two fields: `role` and `content`. There are three roles you will see across this book.

- **`user`** — something the human (or, later, the world) sent in.
- **`assistant`** — something the model said.
- **`system`** — out-of-band instructions that shape who the model is and what it should do. We will get to this in Chapter 3.

A conversation is a list of messages, in the order they happened. The model reads the entire list every time. The API itself has no memory: each `messages.create` call is a stateless function from a list of messages to a reply. That sounds like a limitation, and in some ways it is, but it is also what makes the API easy to reason about. Whatever the model "remembers" about a conversation is exactly the messages list you sent. You can read more about this here [4].

The `content` field, in our example, is just a string. It can also be a list of typed content blocks: text, images, tool calls, tool results. 

## What is a token?

The model sees tokens instead of characters. A token is a chunk of text, usually a word, sometimes a word fragment, sometimes a single character. The string `"Say hello in five words."` is about seven tokens. The string `"antidisestablishmentarianism"` is also seven, despite being one English word. As a rough rule, one token is about four characters of English text.

Tokens matter because:

1. **The model has a context window** — a maximum number of tokens it can see at once. For Claude Opus 4.7 it is one million, which is enormous, but you will hit it eventually if you stuff entire codebases into the prompt. The window covers everything: system prompt, full conversation history, current message, and the model's reply.

2. **Tokens cost money.** Input tokens (what you send) are cheaper than output tokens (what the model generates). Pricing changes all the time so check the provider's site. The relevant point for now is that long conversations cost more than short ones, and the cost grows roughly linearly with the conversation length on each turn. We will revisit this in Chapter 15 when we learn about history compression.

3. **`max_tokens` is a cap on the reply.**  A response that hits the cap will be cut off mid-sentence. 

You can ask the SDK to count tokens for you ahead of time:

```python
client.messages.count_tokens(
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Say hello in five words."}],
).input_tokens
# 12
```

## What is a completion?

The thing `messages.create` returns is, confusingly, also called a message — a `Message` object representing the assistant's reply. (The Anthropic API uses "message" for both directions, while some providers call the response a "completion" or a "choice." We will use the word response in prose to keep things easily readable.)

A response has three things worth knowing about:

```python
>>> response.content
[TextBlock(text='Hello! Welcome, how are you?', type='text')]
>>> response.stop_reason
'end_turn'
>>> response.usage
Usage(input_tokens=12, output_tokens=10, ...)
```

**`content`** is a list of content blocks. For a plain text reply, that list has exactly one block, which is a `TextBlock` object. When tools enter the picture (Chapter 7), the same field can hold `ToolUseBlock`s. Model's thinking will result in a `ThinkingBlock` object. So even though we will mostly be reaching for `response.content[0].text` in this chapter, do not hardcode that pattern in your loop yet. We will write it correctly the first time when we get to Chapter 7.

**`stop_reason`** tells you why the model stopped generating. The values you will encounter most are:

- `end_turn` — the model finished a complete reply. This is the happy path.
- `max_tokens` — the model hit our cap mid-thought. The reply is truncated. Either raise `max_tokens` or stream the response.
- `tool_use` — the model wants to call a tool instead of replying. 
- `refusal` — the model refused on safety grounds.

Chapter 9 will treat stop reasons as first-class control-flow signals. For now, mentally check that you are seeing `end_turn`.

**`usage`** reports how many input and output tokens the call consumed. Multiply by your provider's per-token rate to get cost. In chapter 24 we are going to assemble a dashboard to monitor the usage.

## Production reference

Open [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and scroll to [`chat()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L540) near the bottom. You will see a produciton version of what we just built, which builds a kwargs dict, calls `self._client.messages.create(**kwargs)`, and hands the result back. The handful of lines we ran in `main.py` are the load-bearing core of a 600-line file. Everything else exists to feed those lines the right inputs and to make sense of the outputs.

Three functions in particular are worth tracing once you have written your own version. Each maps directly to something we wrote or discussed in this chapter:

- **[`_build_kwargs()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L416)** assembles the `model`, `messages`, `max_tokens`, `temperature`, and (optionally) `system`, `tools`, and `thinking` parameters from a higher-level request. Today we hardcoded these and Chapter 5 generalizes the assembly so that the same call can target any provider.
- **[`_parse_response()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L485)** is the production version of `response.content[0].text`. Notice that it does not index `[0]` and rather walks the full block list, accumulates every text block, and routes `tool_use` and `thinking` blocks down separate paths. Chapters 7 and 9 build up to writing this loop properly.
- **[`_handle_error()`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L56)** classifies SDK exceptions into a structured response: status code, retry-after, error kind. We will revisit this when we add observability in Chapters 9 and 24.

Two production nuances are worth flagging now, because they are invisible from a toy script but inevitable the moment a real loop wraps the call:

1. The client is `AsyncAnthropic`, not `Anthropic`. An agent that talks to a chat channel while running a heartbeat while streaming a response cannot afford to block on a single network call. The synchronous SDK is fine for our `main.py`.
2. Retries are off at the SDK level (`max_retries=0` in the client constructor) and centralized one layer up. If both layers retried, a transient 429 would be retried twice over — once by the SDK, once by the wrapper — and back-off windows would multiply.

Inside [`nanobot/nanobot/providers/`](https://github.com/HKUDS/nanobot/tree/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers), you will also see one file per LLM backend ([`anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py), [`openai_compat_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/openai_compat_provider.py), [`azure_openai_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/azure_openai_provider.py), [`github_copilot_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/github_copilot_provider.py), etc.) and a [`base.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/base.py) that defines the shared interface. We are going to arrive to this structure by the end of Chapter 5.

## Exercises

1. **Model identity.** Send the prompt `"Without searching the web, what model are you?"` to three different models — `claude-opus-4-6`, `claude-sonnet-4-6`, and `claude-haiku-4-5` — and compare the answers. Models are often unreliable narrators about themselves. This tells you to be critical about the LLM as a source of truth.

2. **Inspect the response object.** Our script reaches for `response.content[0].text` and ignores everything else. Take the same script and ask the model `"Think briefly before answering: what is 17 * 23?"`, but add `thinking={"type": "adaptive"}` as an extra argument to `messages.create`. Then `print(response.content)` raw without indexing and inspect the `type` field on each block. The list now contains more than one block, which is the first concrete hint of why we warned against hardcoding `[0]`. While you are at it, print `response.stop_reason` and `response.usage` and notice that the token cost has gone up: the model is being charged for thoughts you can now read.

3. **Write a robust text extractor.** The trap we keep flagging — `response.content[0].text` — is one line away from being right. Write a helper `extract_text(response) -> str` that iterates `response.content`, keeps only blocks whose `type == "text"`, and joins their `text` fields. Run it against (a) a plain reply and (b) the adaptive-thinking response from Exercise 2. Then open [`_parse_response`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L485) in [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and compare. The shape will be very close to yours.

4. **Stretch: count tokens before sending.** Use `client.messages.count_tokens(...)` to compute the input-token cost of a candidate prompt before you send it. Wrap this into a small helper `cost_estimate(messages, model)` that returns approximate dollars based on a rate you hardcode. We will come back to this in Chapter 24.

5. **Stretch: prompt caching.** A long, repeated prefix — a system prompt, a knowledge document, a tool catalog — is paid for on every single call unless you mark it for caching. Open [`_apply_cache_control`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py#L379) in [`nanobot/nanobot/providers/anthropic_provider.py`](https://github.com/HKUDS/nanobot/blob/28f9bbff314cf90b0401b3aa220ca7a723c4f4ab/nanobot/providers/anthropic_provider.py) and observe how nanobot attaches `"cache_control": {"type": "ephemeral"}` to the last block of the system prompt. Reproduce that yourself: pass a `system` parameter shaped like `[{"type": "text", "text": <a few hundred tokens of fake docs>, "cache_control": {"type": "ephemeral"}}]` and call the model twice with the same system prompt and a different user question each time. Print `response.usage` after each call. The first response will report `cache_creation_input_tokens`; the second should report `cache_read_input_tokens` instead, billed at roughly a tenth of the per-token rate. Prompt caching is one of the cheapest, highest-leverage levers in production and we will lean on it again in Chapter 3 when the system prompt starts growing.


## References

[1] *uv — an extremely fast Python package and project manager.* <https://docs.astral.sh/uv/>

[2] *Anthropic Python SDK reference.* <https://platform.claude.com/docs/en/api/sdks/python>

[3] *API vs. SDK: what's the difference?* IBM Think. <https://www.ibm.com/think/topics/api-vs-sdk>

[4] *Using the Messages API — Multiple conversational turns.* Claude API documentation. <https://platform.claude.com/docs/en/build-with-claude/working-with-messages>
