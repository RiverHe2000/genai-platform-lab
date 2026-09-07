"""Chat-model backends, both speaking the :class:`mcpeval.schemas.ChatModel` protocol.

Two implementations serve two different jobs:

* :class:`ScriptedChatModel` is the model every deterministic test in the project uses.
  A benchmark whose own test suite needs a GPU cannot be trusted or maintained, so the
  agent loop, the permission policy and the grader are all exercised against a model
  whose next reply is decided by the test, not by sampling.
* :class:`HFChatModel` runs a real Hugging Face causal LM in-process for the actual
  benchmark runs, behind a content-addressed SQLite cache so that re-scoring a run, or
  re-running a task after a grader fix, costs nothing and returns byte-identical text.

Token counts come from :func:`estimate_tokens` rather than a tokeniser. The scripted
model has no tokeniser to consult, and the benchmark compares architectures by tokens
spent: what matters is that the number is deterministic and the same yardstick is
applied to every architecture, not that it matches any particular vendor's billing.

`torch` and `transformers` are imported lazily and defensively, so `import
mcpeval.agents.llm` stays cheap and works in a CI job that installs neither.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from pydantic import ValidationError

from mcpeval.schemas import Completion, Message, Usage

__all__ = [
    "ChatModelError",
    "HFChatModel",
    "MissingDependencyError",
    "Responder",
    "ResponseCache",
    "ScriptExhaustedError",
    "ScriptedCall",
    "ScriptedChatModel",
    "cache_key",
    "estimate_message_tokens",
    "estimate_tokens",
]

_CHARS_PER_TOKEN: Final = 4
"""Bytes per token assumed by :func:`estimate_tokens`; roughly right for English."""

_MESSAGE_OVERHEAD_TOKENS: Final = 4
"""Tokens charged per message for the chat template's role framing."""


class ChatModelError(RuntimeError):
    """Base class for the failures raised by this module."""


class ScriptExhaustedError(ChatModelError):
    """A :class:`ScriptedChatModel` was asked for a reply it was never given.

    This is deliberately loud. A scripted model that silently returned an empty
    string would let an agent test pass while the agent took a completely different
    path through the conversation than the test author intended.
    """


class MissingDependencyError(ChatModelError):
    """An optional dependency (`torch`, `transformers`) is not installed."""


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in `text` without loading a tokeniser.

    The estimator is ``max(words, ceil(len(stripped) / 4))`` over the whitespace-stripped
    text: the word count is the floor (a byte-pair tokeniser rarely merges across a
    space), and one token per four characters catches long identifiers, JSON blobs and
    non-English text that split into far more tokens than they have words.

    Args:
        text: Any string; leading and trailing whitespace is ignored.

    Returns:
        A non-negative token estimate. Zero exactly when `text` is empty or blank.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    words = len(stripped.split())
    chars = -(-len(stripped) // _CHARS_PER_TOKEN)
    return max(words, chars)


def estimate_message_tokens(
    messages: Sequence[Message], *, overhead: int = _MESSAGE_OVERHEAD_TOKENS
) -> int:
    """Estimate the prompt size of a whole conversation.

    Args:
        messages: The conversation as it would be handed to a model.
        overhead: Tokens charged per message for the role and name framing a chat
            template adds around the content. Pass ``0`` to count content only.

    Returns:
        The summed estimate over every message.
    """
    return sum(estimate_tokens(message.content) + overhead for message in messages)


def _truncate_at_stop(text: str, stop: Sequence[str] | None) -> tuple[str, bool]:
    """Cut `text` at the earliest stop sequence, returning the text and whether it hit."""
    if not stop:
        return text, False
    cuts = [text.index(marker) for marker in stop if marker and marker in text]
    if not cuts:
        return text, False
    return text[: min(cuts)], True


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Cut `text` down to roughly `max_tokens` tokens, returning whether it was cut."""
    if estimate_tokens(text) <= max_tokens:
        return text, False
    return text[: max(0, max_tokens) * _CHARS_PER_TOKEN].rstrip(), True


# --------------------------------------------------------------------------------------
# The scripted model
# --------------------------------------------------------------------------------------

Responder = Callable[[Sequence[Message]], str]
"""A callable that decides a reply from the whole conversation."""


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    """One call a :class:`ScriptedChatModel` served, kept so tests can assert on it."""

    messages: tuple[Message, ...]
    reply: str
    source: str
    """Which strategy produced the reply: ``route``, ``queue``, ``responder`` or ``default``."""
    max_tokens: int
    temperature: float
    stop: tuple[str, ...]
    usage: Usage

    @property
    def last_message(self) -> Message | None:
        """The final message of the prompt, or `None` for an empty conversation."""
        return self.messages[-1] if self.messages else None

    @property
    def system_prompt(self) -> str:
        """The first system message, or the empty string when there was none."""
        return next((m.content for m in self.messages if m.role == "system"), "")


@dataclass
class ScriptedChatModel:
    """A deterministic, programmable stand-in for a language model.

    Replies are resolved by a fixed cascade, so one instance can mix the three styles
    without ambiguity:

    1. **routes** - the first route whose key occurs (case-insensitively) in the last
       user or tool message wins. Use this for "whenever it asks about fees, say this",
       which keeps a test readable when the agent's step order is not the point.
    2. **replies** - the next canned reply is popped off a FIFO queue. Use this for
       "turn one, then turn two", the common case for agent-loop tests.
    3. **responder** - a callable is handed the whole conversation. Use this when the
       reply has to depend on what the tools returned.
    4. **default** - a constant fallback, if one was given.

    With none of those left, :class:`ScriptExhaustedError` is raised rather than a silent
    empty reply, because running past the end of a script means the agent took a
    different path than the test describes, and that is exactly what a test should catch.

    Attributes:
        calls: Every call served, in order, with the messages that were seen.
        usage: Running total of the estimated tokens across all calls.

    Example:
        >>> model = ScriptedChatModel(["first", "second"], default="fallback")
        >>> model.complete([Message(role="user", content="hello")]).text
        'first'
    """

    replies: Sequence[str] | None = None
    routes: Mapping[str, str] | Sequence[tuple[str, str]] | None = None
    responder: Responder | None = None
    default: str | None = None
    model_name: str = "scripted"
    calls: list[ScriptedCall] = field(default_factory=list, init=False)
    usage: Usage = field(default=Usage(), init=False)

    def __post_init__(self) -> None:
        self._scripted: tuple[str, ...] = tuple(self.replies or ())
        self._queue: list[str] = list(self._scripted)
        self._routes: list[tuple[str, str]] = list(
            self.routes.items() if isinstance(self.routes, Mapping) else (self.routes or ())
        )

    @property
    def name(self) -> str:
        """Identifier recorded in the trajectory."""
        return self.model_name

    @property
    def call_count(self) -> int:
        """How many completions have been served."""
        return len(self.calls)

    @property
    def remaining(self) -> int:
        """How many canned replies are still queued."""
        return len(self._queue)

    @property
    def last_call(self) -> ScriptedCall | None:
        """The most recent call, or `None` if the model has not been used."""
        return self.calls[-1] if self.calls else None

    @property
    def last_messages(self) -> tuple[Message, ...]:
        """The conversation seen on the most recent call, empty if there was none."""
        return self.calls[-1].messages if self.calls else ()

    def push(self, *replies: str) -> ScriptedChatModel:
        """Append more canned replies to the queue and return self, for chaining."""
        self._queue.extend(replies)
        return self

    def route(self, needle: str, reply: str) -> ScriptedChatModel:
        """Add a routing rule and return self, for chaining.

        Args:
            needle: Matched case-insensitively against the last user or tool message.
            reply: The text to return when it matches.
        """
        self._routes.append((needle, reply))
        return self

    def reset(self) -> ScriptedChatModel:
        """Restore the original reply queue and forget every recorded call."""
        self._queue = list(self._scripted)
        self.calls.clear()
        self.usage = Usage()
        return self

    @staticmethod
    def _match_target(messages: Sequence[Message]) -> str | None:
        """The text routes are matched against: the last user or tool turn."""
        for message in reversed(messages):
            if message.role in {"user", "tool"}:
                return message.content
        return messages[-1].content if messages else None

    def _resolve(self, messages: Sequence[Message]) -> tuple[str, str]:
        target = self._match_target(messages)
        if target is not None:
            lowered = target.lower()
            for needle, reply in self._routes:
                if needle.lower() in lowered:
                    return reply, "route"
        if self._queue:
            return self._queue.pop(0), "queue"
        if self.responder is not None:
            return self.responder(messages), "responder"
        if self.default is not None:
            return self.default, "default"
        raise ScriptExhaustedError(self._exhausted_message(target))

    def _exhausted_message(self, target: str | None) -> str:
        preview = "" if target is None else target[:120]
        return (
            f"{self.model_name!r} ran out of scripted replies on call {len(self.calls) + 1}: "
            f"{len(self._scripted)} replies were scripted and all are used, "
            f"{len(self._routes)} route(s) failed to match, "
            f"and no responder or default was configured. Last message: {preview!r}"
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        """Return the next scripted reply.

        `stop` and `max_tokens` are honoured against the estimated token count so that
        agent code handling truncated output is exercised by the scripted model too.

        Args:
            messages: The conversation so far; recorded on :attr:`calls`.
            max_tokens: Replies longer than this estimate are cut, giving
                ``finish_reason == "length"``.
            temperature: Recorded only; the scripted model is always deterministic.
            stop: Sequences at which the reply is cut.

        Returns:
            The completion, with an estimated :class:`~mcpeval.schemas.Usage`.

        Raises:
            ScriptExhaustedError: If no strategy can produce a reply.
        """
        reply, source = self._resolve(messages)
        text, stopped = _truncate_at_stop(reply, stop)
        text, cut = _truncate_to_tokens(text, max_tokens)
        usage = Usage(
            prompt_tokens=estimate_message_tokens(messages),
            completion_tokens=estimate_tokens(text),
        )
        self.usage = self.usage + usage
        self.calls.append(
            ScriptedCall(
                messages=tuple(messages),
                reply=text,
                source=source,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=tuple(stop or ()),
                usage=usage,
            )
        )
        return Completion(
            text=text,
            usage=usage,
            finish_reason="length" if cut and not stopped else "stop",
        )


# --------------------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------------------


def cache_key(
    model: str,
    messages: Sequence[Message],
    *,
    max_tokens: int,
    temperature: float,
    stop: Sequence[str] | None = None,
) -> str:
    """Content-address one model call.

    The key covers everything that can change the answer - the model, every message
    including the role and the tool name that produced it, and the decoding parameters -
    so a hit is a genuine repeat of the same question and never a near miss.

    Args:
        model: The model identifier.
        messages: The full conversation handed to the model.
        max_tokens: Decoding limit.
        temperature: Decoding temperature.
        stop: Stop sequences, if any.

    Returns:
        A hex SHA-256 digest, stable across processes and platforms.
    """
    payload = {
        "model": model,
        "messages": [[m.role, m.name or "", m.content] for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": list(stop or ()),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """A SQLite cache of completions, keyed by :func:`cache_key`.

    SQLite rather than a directory of files because a benchmark run makes thousands of
    small calls across several processes, and one file that survives a crash mid-run is
    easier to reason about than a tree of partial writes.

    Attributes:
        hits: Lookups that were served from the cache.
        misses: Lookups that were not, including entries that failed to deserialise.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        """Open (and create if needed) the cache at `path`.

        Args:
            path: A file path, or ``":memory:"`` for a process-local cache.
        """
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS completions ("
            "key TEXT PRIMARY KEY, model TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Completion | None:
        """Return the cached completion for `key`, or `None` on a miss.

        A row that no longer deserialises (the schema moved under an old cache file) is
        deleted and reported as a miss rather than raising: a stale cache must never be
        able to break a run.
        """
        row = self._conn.execute("SELECT payload FROM completions WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        try:
            completion = Completion.model_validate_json(row[0])
        except ValidationError:
            self._conn.execute("DELETE FROM completions WHERE key = ?", (key,))
            self._conn.commit()
            self.misses += 1
            return None
        self.hits += 1
        return completion

    def put(self, key: str, model: str, completion: Completion) -> None:
        """Store `completion` under `key`, replacing any existing row."""
        self._conn.execute(
            "INSERT OR REPLACE INTO completions (key, model, payload) VALUES (?, ?, ?)",
            (key, model, completion.model_dump_json()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        """The number of cached completions."""
        row = self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# --------------------------------------------------------------------------------------
# The Hugging Face model
# --------------------------------------------------------------------------------------


def _optional_import(name: str) -> Any | None:
    """Import `name`, returning `None` when it is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


@contextmanager
def _inference_mode() -> Iterator[None]:
    """Disable autograd when torch is present.

    This is a memory optimisation, not a correctness requirement, so when torch is
    absent - which happens in tests that drive the generation path with an injected
    fake model - there is simply nothing to switch off.
    """
    torch = _optional_import("torch")
    if torch is None:
        yield
        return
    with torch.inference_mode():
        yield


def _resolve_device(torch: Any | None, spec: str) -> str:
    """Turn a device spec into a concrete device.

    Args:
        torch: The torch module, or `None` if it is not installed.
        spec: ``"auto"``, or any device string to use verbatim.

    Returns:
        ``"cuda"`` when `spec` is ``"auto"`` and a CUDA device is available, otherwise
        ``"cpu"``; any other `spec` unchanged.
    """
    if spec != "auto":
        return spec
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(torch: Any, spec: str, device: str) -> Any:
    """Turn a dtype spec into a torch dtype.

    Args:
        torch: The torch module.
        spec: ``"auto"`` or the name of a torch dtype, e.g. ``"bfloat16"``.
        device: The resolved device; ``"auto"`` picks bfloat16 on CUDA and float32
            elsewhere, because float16 on CPU is slower than float32 and bfloat16 on an
            older GPU is not supported.

    Returns:
        The torch dtype object.

    Raises:
        ChatModelError: If `spec` names no torch dtype.
    """
    if spec == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    dtype = getattr(torch, spec, None)
    if dtype is None:
        raise ChatModelError(f"unknown dtype {spec!r}: expected 'auto' or a torch dtype name")
    return dtype


class HFChatModel:
    """A Hugging Face causal LM behind the :class:`~mcpeval.schemas.ChatModel` protocol.

    Weights are loaded on the first :meth:`complete`, not in ``__init__``, so that
    constructing a benchmark configuration - which happens in argument parsing, in the
    CLI's ``--help``, and in every test that merely names a model - never costs a
    multi-gigabyte load.

    Decoding is greedy at ``temperature == 0``, and only then is the answer cached: at
    any higher temperature the reply is a sample rather than a function of the key, and
    caching it would quietly freeze one draw and call it reproducibility.

    Attributes:
        cache: The response cache, or `None` when caching is disabled.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        cache_path: str | Path | None = None,
        cache: ResponseCache | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        seed: int = 0,
    ) -> None:
        """Configure the model without loading anything.

        Args:
            model_name: A Hugging Face repository id, e.g. ``Qwen/Qwen2.5-1.5B-Instruct``.
            device: ``"auto"``, ``"cpu"``, ``"cuda"``, or any torch device string.
            dtype: ``"auto"`` or a torch dtype name such as ``"bfloat16"``.
            cache_path: Where to keep the response cache; `None` disables caching unless
                `cache` is given.
            cache: An already-open cache to share between models.
            model: A pre-built model, which bypasses loading entirely. Tests inject a
                fake here to exercise the generation path without weights.
            tokenizer: A pre-built tokeniser, as above.
            seed: Seed applied before sampling when ``temperature > 0``.
        """
        self._model_name = model_name
        self._device_spec = device
        self._dtype_spec = dtype
        self._model = model
        self._tokenizer = tokenizer
        self._device: str | None = None
        self._seed = seed
        self._owns_cache = cache is None and cache_path is not None
        self.cache = (
            cache if cache is not None else (ResponseCache(cache_path) if cache_path else None)
        )

    @property
    def name(self) -> str:
        """Identifier recorded in the trajectory."""
        return self._model_name

    @property
    def loaded(self) -> bool:
        """Whether the weights are in memory yet."""
        return self._model is not None and self._tokenizer is not None

    def close(self) -> None:
        """Close the cache if this model opened it."""
        if self._owns_cache and self.cache is not None:
            self.cache.close()

    def _ensure_loaded(self) -> tuple[Any, Any, str]:
        """Load the weights on first use and return ``(model, tokenizer, device)``."""
        torch = _optional_import("torch")
        if self._device is None:
            self._device = _resolve_device(torch, self._device_spec)
        if self._model is None or self._tokenizer is None:
            transformers = _optional_import("transformers")
            if torch is None or transformers is None:
                raise MissingDependencyError(
                    "HFChatModel needs torch and transformers; install the optional "
                    "dependencies with: pip install 'mcpeval[hf]'"
                )
            dtype = _resolve_dtype(torch, self._dtype_spec, self._device)
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self._model_name, dtype=dtype
            )
            self._model.to(self._device)
            self._model.eval()
        return self._model, self._tokenizer, self._device

    @staticmethod
    def _to_chat(messages: Sequence[Message]) -> list[dict[str, str]]:
        """Render the conversation in the shape `apply_chat_template` expects.

        Tool results are rendered as user turns labelled with the tool name. Small
        instruct models' chat templates either reject a bare ``tool`` role or drop it
        (it is only valid after an assistant ``tool_calls`` block they never emitted),
        and losing a tool result silently is far worse than labelling it by hand: the
        action protocol in :mod:`mcpeval.agents.protocol` carries the structure anyway.
        """
        chat: list[dict[str, str]] = []
        for message in messages:
            if message.role == "tool":
                label = message.name or "tool"
                chat.append({"role": "user", "content": f"[{label} result]\n{message.content}"})
            else:
                chat.append({"role": message.role, "content": message.content})
        return chat

    @staticmethod
    def _flatten(messages: Sequence[Message]) -> str:
        """Fallback prompt for a base model with no chat template."""
        lines = [f"{m.role.upper()}: {m.content}" for m in messages]
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _encode(self, tokenizer: Any, messages: Sequence[Message], device: str) -> dict[str, Any]:
        """Tokenise the conversation, using the chat template when the model has one."""
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                self._to_chat(messages),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(self._flatten(messages), return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        stop: Sequence[str] | None,
    ) -> Completion:
        """Run the model once and package the result."""
        model, tokenizer, device = self._ensure_loaded()
        inputs = self._encode(tokenizer, messages, device)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": pad_id
            if pad_id is not None
            else getattr(tokenizer, "eos_token_id", None),
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            torch = _optional_import("torch")
            if torch is not None:
                torch.manual_seed(self._seed)
        with _inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
        prompt_tokens = len(inputs["input_ids"][0])
        new_tokens = output[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text, stopped = _truncate_at_stop(text, stop)
        completion_tokens = len(new_tokens)
        finish = "length" if completion_tokens >= max_tokens and not stopped else "stop"
        return Completion(
            text=text.strip(),
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason=finish,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        """Generate the assistant's next message.

        Args:
            messages: The conversation so far.
            max_tokens: Maximum new tokens to generate.
            temperature: ``0.0`` means greedy decoding, and only then is the cache used.
            stop: Sequences at which to cut the decoded text.

        Returns:
            The completion, with token counts taken from the tokeniser rather than
            estimated.
        """
        key = cache_key(
            self._model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        cache = self.cache if temperature == 0.0 else None
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        completion = self._generate(
            messages, max_tokens=max_tokens, temperature=temperature, stop=stop
        )
        if cache is not None:
            cache.put(key, self._model_name, completion)
        return completion
