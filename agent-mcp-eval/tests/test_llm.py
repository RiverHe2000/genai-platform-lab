"""Tests for the chat-model backends.

The scripted model is the backbone of every other deterministic test in the project, so
it is tested harder than its size suggests: a bug in the fake model would show up as a
confusing failure somewhere in the agent loop rather than here.

The Hugging Face model is exercised through injected fakes. That covers everything the
class actually owns - the cache decision, the chat template, the stop and length
handling - without a download, and leaves only the two `from_pretrained` lines untested.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mcpeval.agents import llm
from mcpeval.agents.llm import (
    ChatModelError,
    HFChatModel,
    MissingDependencyError,
    ResponseCache,
    ScriptedChatModel,
    ScriptExhaustedError,
    _resolve_device,
    _resolve_dtype,
    cache_key,
    estimate_message_tokens,
    estimate_tokens,
)
from mcpeval.schemas import ChatModel, Completion, Message, Usage

TEXT = st.text(alphabet=st.characters(codec="utf-8"), max_size=200)


def _user(content: str) -> Message:
    return Message(role="user", content=content)


def _conversation() -> list[Message]:
    return [
        Message(role="system", content="You are the supervisor agent."),
        _user("What is ACC-0001 worth?"),
        Message(role="tool", name="account_valuation", content="412300.50"),
    ]


def _no_optional_imports(name: str) -> Any | None:
    """Stand-in for `llm._optional_import` reporting every optional package as absent."""
    return None if name in {"torch", "transformers"} else importlib.import_module(name)


def _takes_chat_model(model: ChatModel) -> str:
    """Static conformance check: mypy rejects this call if the protocol is not met."""
    return model.name


# --------------------------------------------------------------------------------------
# estimate_tokens
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", " ", "\n\t  \r\n"])
def test_blank_text_costs_no_tokens(blank: str) -> None:
    assert estimate_tokens(blank) == 0


def test_word_count_is_the_floor() -> None:
    # Six short words are only eleven characters, so the word count sets the price.
    assert estimate_tokens("a b c d e f") == 6


def test_dense_text_is_priced_by_characters() -> None:
    # One 40-character "word" is one word but ten tokens' worth of characters.
    assert estimate_tokens("a" * 40) == 10


@given(TEXT)
def test_estimate_is_deterministic_and_ignores_outer_whitespace(text: str) -> None:
    assert estimate_tokens(text) == estimate_tokens(text)
    assert estimate_tokens(text) == estimate_tokens(text.strip())
    assert estimate_tokens(text) >= 0


@given(TEXT, TEXT)
def test_estimate_is_monotone_under_concatenation(left: str, right: str) -> None:
    """Adding text can never make a prompt cheaper: the metric would be gameable."""
    joined = estimate_tokens(left + right)
    assert joined >= estimate_tokens(left)
    assert joined >= estimate_tokens(right)


def test_message_tokens_add_the_role_overhead() -> None:
    messages = [_user("a b"), _user("c")]
    assert estimate_message_tokens(messages, overhead=0) == 3
    assert estimate_message_tokens(messages, overhead=5) == 13


def test_message_tokens_of_an_empty_conversation() -> None:
    assert estimate_message_tokens([]) == 0


# --------------------------------------------------------------------------------------
# ScriptedChatModel
# --------------------------------------------------------------------------------------


def test_scripted_model_satisfies_the_chat_model_protocol() -> None:
    model = ScriptedChatModel(["hi"], model_name="scripted-1")
    assert isinstance(model, ChatModel)
    assert _takes_chat_model(model) == "scripted-1"


def test_replies_are_served_in_order() -> None:
    model = ScriptedChatModel(["first", "second"])
    assert model.complete([_user("a")]).text == "first"
    assert model.remaining == 1
    assert model.complete([_user("b")]).text == "second"
    assert model.remaining == 0
    assert model.call_count == 2


def test_running_out_of_script_raises_a_message_that_says_where() -> None:
    model = ScriptedChatModel(["only one"], model_name="agent-model")
    model.complete([_user("a")])
    with pytest.raises(ScriptExhaustedError) as excinfo:
        model.complete([_user("the unexpected second turn")])
    message = str(excinfo.value)
    assert "agent-model" in message
    assert "call 2" in message
    assert "the unexpected second turn" in message


def test_routes_match_the_last_user_message() -> None:
    model = ScriptedChatModel(routes={"fee": "about fees", "price": "about prices"})
    assert model.complete([_user("what is the price?")]).text == "about prices"
    assert model.last_call is not None
    assert model.last_call.source == "route"


def test_routes_match_a_tool_result_over_an_earlier_user_turn() -> None:
    model = ScriptedChatModel(routes={"412300": "the valuation came back"})
    assert model.complete(_conversation()).text == "the valuation came back"


def test_routes_are_case_insensitive() -> None:
    model = ScriptedChatModel(routes={"FEE Schedule": "matched"})
    assert model.complete([_user("show me the fee schedule please")]).text == "matched"


def test_routes_look_past_the_agents_own_last_turn() -> None:
    model = ScriptedChatModel(routes={"my question": "matched"})
    messages = [_user("here is my question"), Message(role="assistant", content="thinking")]
    assert model.complete(messages).text == "matched"


def test_a_conversation_with_no_user_turn_matches_on_its_last_message() -> None:
    model = ScriptedChatModel(routes={"supervisor": "matched"})
    system = Message(role="system", content="You are the supervisor agent.")
    assert model.complete([system]).text == "matched"


def test_the_first_matching_route_wins() -> None:
    model = ScriptedChatModel(routes=[("fee", "first"), ("fee schedule", "second")])
    assert model.complete([_user("fee schedule")]).text == "first"


def test_a_route_takes_precedence_over_the_queue() -> None:
    model = ScriptedChatModel(["queued"], routes={"urgent": "routed"})
    assert model.complete([_user("urgent request")]).text == "routed"
    assert model.remaining == 1, "a routed reply must not consume the queue"


def test_the_responder_sees_the_whole_conversation() -> None:
    def responder(messages: Sequence[Message]) -> str:
        return f"{len(messages)} messages, last from {messages[-1].role}"

    model = ScriptedChatModel(responder=responder)
    assert model.complete(_conversation()).text == "3 messages, last from tool"
    assert model.last_call is not None
    assert model.last_call.source == "responder"


def test_the_default_is_the_last_resort() -> None:
    model = ScriptedChatModel(["queued"], default="fallback")
    assert model.complete([_user("a")]).text == "queued"
    assert model.complete([_user("b")]).text == "fallback"
    assert model.complete([_user("c")]).text == "fallback"


def test_an_empty_conversation_falls_through_to_the_queue() -> None:
    model = ScriptedChatModel(["queued"], routes={"never": "routed"})
    assert model.complete([]).text == "queued"


def test_every_call_records_the_conversation_it_saw() -> None:
    model = ScriptedChatModel(["a", "b"])
    first = _conversation()
    model.complete(first)
    model.complete([_user("second turn")])
    assert model.calls[0].messages == tuple(first)
    assert model.calls[0].system_prompt == "You are the supervisor agent."
    assert model.calls[0].last_message == first[-1]
    assert model.last_messages == (_user("second turn"),)


def test_an_unused_model_has_nothing_to_report() -> None:
    model = ScriptedChatModel(["a"])
    assert model.last_call is None
    assert model.last_messages == ()


def test_a_conversation_with_no_system_message_has_an_empty_system_prompt() -> None:
    model = ScriptedChatModel(["a"])
    model.complete([_user("hi")])
    assert model.calls[0].system_prompt == ""


def test_usage_is_the_sum_of_the_calls() -> None:
    """Conservation law: the running total is exactly what the calls reported."""
    model = ScriptedChatModel(["one two three", "four"])
    model.complete([_user("a longer prompt here")])
    model.complete(_conversation())
    total = Usage()
    for call in model.calls:
        total = total + call.usage
    assert model.usage == total
    assert model.usage.total_tokens > 0


def test_recorded_decoding_parameters_survive_on_the_call() -> None:
    model = ScriptedChatModel(["reply"])
    model.complete([_user("a")], max_tokens=99, temperature=0.7, stop=["\n"])
    call = model.calls[0]
    assert (call.max_tokens, call.temperature, call.stop) == (99, 0.7, ("\n",))


def test_a_stop_sequence_cuts_the_reply() -> None:
    model = ScriptedChatModel(["answer\nOBSERVATION: leaked"])
    completion = model.complete([_user("a")], stop=["\nOBSERVATION:"])
    assert completion.text == "answer"
    assert completion.finish_reason == "stop"


def test_an_absent_stop_sequence_changes_nothing() -> None:
    model = ScriptedChatModel(["answer"])
    assert model.complete([_user("a")], stop=["<|im_end|>"]).text == "answer"


def test_max_tokens_truncates_and_reports_length() -> None:
    model = ScriptedChatModel(["one two three four five six seven eight"])
    completion = model.complete([_user("a")], max_tokens=3)
    assert completion.finish_reason == "length"
    assert estimate_tokens(completion.text) <= 3
    assert completion.usage.completion_tokens <= 3


def test_push_and_route_chain_and_extend_the_script() -> None:
    model = ScriptedChatModel().push("one").route("later", "routed").push("two")
    assert model.complete([_user("a")]).text == "one"
    assert model.complete([_user("later on")]).text == "routed"
    assert model.complete([_user("b")]).text == "two"


def test_reset_restores_the_original_script() -> None:
    model = ScriptedChatModel(["one", "two"])
    model.complete([_user("a")])
    model.reset()
    assert model.remaining == 2
    assert model.calls == []
    assert model.usage == Usage()
    assert model.complete([_user("a")]).text == "one"


def test_pushed_replies_are_dropped_by_reset() -> None:
    model = ScriptedChatModel(["one"]).push("extra")
    model.reset()
    assert model.remaining == 1


# --------------------------------------------------------------------------------------
# Cache key and cache
# --------------------------------------------------------------------------------------


def test_the_cache_key_is_stable_for_the_same_call() -> None:
    args: dict[str, Any] = {"max_tokens": 128, "temperature": 0.0, "stop": ["\n"]}
    first = cache_key("m", _conversation(), **args)
    second = cache_key("m", list(_conversation()), **args)
    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model", "other-model"),
        ("max_tokens", 256),
        ("temperature", 0.7),
        ("stop", ["<end>"]),
    ],
)
def test_the_cache_key_changes_when_anything_that_matters_changes(
    field_name: str, value: Any
) -> None:
    base: dict[str, Any] = {
        "model": "m",
        "max_tokens": 128,
        "temperature": 0.0,
        "stop": ["\n"],
    }
    changed = {**base, field_name: value}
    original = cache_key(base.pop("model"), _conversation(), **base)
    assert cache_key(changed.pop("model"), _conversation(), **changed) != original


def test_the_cache_key_changes_when_a_message_changes() -> None:
    base = _conversation()
    edited = [*base[:-1], Message(role="tool", name="account_valuation", content="0.00")]
    renamed = [*base[:-1], Message(role="tool", name="other_tool", content="412300.50")]
    key = cache_key("m", base, max_tokens=1, temperature=0.0)
    assert cache_key("m", edited, max_tokens=1, temperature=0.0) != key
    assert cache_key("m", renamed, max_tokens=1, temperature=0.0) != key


def test_the_cache_round_trips_a_completion_and_counts_hits() -> None:
    completion = Completion(text="hi", usage=Usage(prompt_tokens=3, completion_tokens=1))
    with ResponseCache() as cache:
        assert cache.get("k") is None
        cache.put("k", "m", completion)
        assert cache.get("k") == completion
        assert (cache.hits, cache.misses) == (1, 1)
        assert len(cache) == 1


def test_the_cache_survives_being_reopened(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "responses.sqlite"
    completion = Completion(text="cached")
    with ResponseCache(path) as writer:
        writer.put("k", "m", completion)
    with ResponseCache(path) as reader:
        assert reader.get("k") == completion


def test_a_row_that_no_longer_deserialises_is_a_miss_not_a_crash() -> None:
    with ResponseCache() as cache:
        cache._conn.execute(
            "INSERT INTO completions (key, model, payload) VALUES (?, ?, ?)",
            ("k", "m", '{"text": 17, "finish_reason": "nonsense"}'),
        )
        assert cache.get("k") is None
        assert cache.misses == 1
        assert len(cache) == 0, "the poisoned row must be evicted"


def test_putting_the_same_key_twice_replaces_the_row() -> None:
    with ResponseCache() as cache:
        cache.put("k", "m", Completion(text="old"))
        cache.put("k", "m", Completion(text="new"))
        assert len(cache) == 1
        cached = cache.get("k")
        assert cached is not None
        assert cached.text == "new"


# --------------------------------------------------------------------------------------
# HFChatModel, driven by fakes
# --------------------------------------------------------------------------------------


class _FakeTensor:
    """The slice of the torch tensor API that :class:`HFChatModel` actually touches."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.device: str | None = None

    def to(self, device: str) -> _FakeTensor:
        self.device = device
        return self

    def __getitem__(self, index: int) -> list[int]:
        return self.rows[index]


class _FakeTokenizer:
    """Records what it was asked to encode and decode, and returns fixed token ids."""

    def __init__(self, *, chat_template: str | None = "{{ messages }}", reply: str = "hi") -> None:
        self.chat_template = chat_template
        self.pad_token_id: int | None = 0
        self.eos_token_id = 7
        self.reply = reply
        self.chats: list[list[dict[str, str]]] = []
        self.template_kwargs: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self.decoded: list[list[int]] = []

    def apply_chat_template(
        self, chat: Sequence[dict[str, str]], **kwargs: Any
    ) -> dict[str, _FakeTensor]:
        self.chats.append([dict(turn) for turn in chat])
        self.template_kwargs.append(dict(kwargs))
        return {"input_ids": _FakeTensor([[1, 2, 3]]), "attention_mask": _FakeTensor([[1, 1, 1]])}

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, _FakeTensor]:
        self.prompts.append(prompt)
        self.template_kwargs.append(dict(kwargs))
        return {"input_ids": _FakeTensor([[1, 2]])}

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        self.decoded.append(list(tokens))
        self.template_kwargs.append(dict(kwargs))
        return self.reply


class _FakeModel:
    def __init__(self, new_tokens: int = 3) -> None:
        self.new_tokens = new_tokens
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> _FakeTensor:
        self.calls.append(kwargs)
        prompt_length = len(kwargs["input_ids"][0])
        return _FakeTensor([[0] * (prompt_length + self.new_tokens)])


def _fake_hf(
    *,
    reply: str = "hi",
    new_tokens: int = 3,
    chat_template: str | None = "{{ messages }}",
    cache: ResponseCache | None = None,
) -> tuple[HFChatModel, _FakeModel, _FakeTokenizer]:
    model = _FakeModel(new_tokens=new_tokens)
    tokenizer = _FakeTokenizer(chat_template=chat_template, reply=reply)
    chat = HFChatModel("fake/model", device="cpu", model=model, tokenizer=tokenizer, cache=cache)
    return chat, model, tokenizer


def test_hf_model_satisfies_the_chat_model_protocol() -> None:
    chat, _, _ = _fake_hf()
    assert isinstance(chat, ChatModel)
    assert _takes_chat_model(chat) == "fake/model"


def test_an_injected_model_counts_as_loaded() -> None:
    chat, _, _ = _fake_hf()
    assert chat.loaded is True
    assert HFChatModel("fake/model").loaded is False


def test_the_chat_template_receives_the_conversation_with_tool_turns_relabelled() -> None:
    chat, _, tokenizer = _fake_hf()
    chat.complete(_conversation())
    rendered = tokenizer.chats[0]
    assert [turn["role"] for turn in rendered] == ["system", "user", "user"]
    assert rendered[-1]["content"].startswith("[account_valuation result]")
    assert "412300.50" in rendered[-1]["content"]


def test_an_unnamed_tool_turn_is_still_labelled() -> None:
    chat, _, tokenizer = _fake_hf()
    chat.complete([Message(role="tool", content="42")])
    assert tokenizer.chats[0][0]["content"] == "[tool result]\n42"


def test_a_model_without_a_chat_template_gets_a_flat_prompt() -> None:
    chat, _, tokenizer = _fake_hf(chat_template=None)
    chat.complete([_user("hello")])
    assert tokenizer.chats == []
    assert tokenizer.prompts[0] == "USER: hello\n\nASSISTANT:"


def test_inputs_are_moved_to_the_device() -> None:
    chat, model, _ = _fake_hf()
    chat.complete([_user("hello")])
    assert model.calls[0]["input_ids"].device == "cpu"
    assert model.calls[0]["attention_mask"].device == "cpu"


def test_greedy_by_default_and_sampling_only_when_asked() -> None:
    chat, model, _ = _fake_hf()
    chat.complete([_user("hello")])
    assert model.calls[0]["do_sample"] is False
    assert "temperature" not in model.calls[0]
    assert model.calls[0]["pad_token_id"] == 0
    chat.complete([_user("hello")], temperature=0.8)
    assert model.calls[1]["do_sample"] is True
    assert model.calls[1]["temperature"] == 0.8


def test_the_end_of_sequence_token_pads_when_there_is_no_pad_token() -> None:
    chat, model, tokenizer = _fake_hf()
    tokenizer.pad_token_id = None
    chat.complete([_user("hello")])
    assert model.calls[0]["pad_token_id"] == 7


def test_token_counts_come_from_the_tokeniser() -> None:
    chat, _, tokenizer = _fake_hf(new_tokens=4)
    completion = chat.complete([_user("hello")], max_tokens=64)
    assert completion.usage == Usage(prompt_tokens=3, completion_tokens=4)
    assert completion.finish_reason == "stop"
    assert tokenizer.decoded == [[0, 0, 0, 0]], "only the new tokens are decoded"
    assert tokenizer.template_kwargs[0]["add_generation_prompt"] is True


def test_hitting_the_token_limit_is_reported_as_length() -> None:
    chat, _, _ = _fake_hf(new_tokens=5)
    assert chat.complete([_user("hello")], max_tokens=5).finish_reason == "length"


def test_a_stop_sequence_cuts_the_decoded_text() -> None:
    chat, _, _ = _fake_hf(reply="the answer\nUSER: next question", new_tokens=9)
    completion = chat.complete([_user("hello")], max_tokens=9, stop=["\nUSER:"])
    assert completion.text == "the answer"
    assert completion.finish_reason == "stop", "a stop hit is not a length cut"


def test_a_completion_is_reused_at_temperature_zero() -> None:
    with ResponseCache() as cache:
        chat, model, _ = _fake_hf(cache=cache)
        first = chat.complete([_user("hello")])
        second = chat.complete([_user("hello")])
        assert first == second
        assert len(model.calls) == 1, "the second call must come from the cache"
        assert (cache.hits, cache.misses) == (1, 1)


def test_a_different_prompt_is_a_different_cache_entry() -> None:
    with ResponseCache() as cache:
        chat, model, _ = _fake_hf(cache=cache)
        chat.complete([_user("hello")])
        chat.complete([_user("goodbye")])
        assert len(model.calls) == 2
        assert len(cache) == 2


def test_sampled_completions_are_never_cached() -> None:
    with ResponseCache() as cache:
        chat, model, _ = _fake_hf(cache=cache)
        chat.complete([_user("hello")], temperature=0.9)
        chat.complete([_user("hello")], temperature=0.9)
        assert len(model.calls) == 2
        assert len(cache) == 0


def test_caching_is_off_when_no_cache_was_configured() -> None:
    chat, model, _ = _fake_hf()
    chat.complete([_user("hello")])
    chat.complete([_user("hello")])
    assert chat.cache is None
    assert len(model.calls) == 2


def test_a_model_closes_only_the_cache_it_opened(tmp_path: Path) -> None:
    owned = HFChatModel("fake/model", cache_path=tmp_path / "c.sqlite")
    assert owned.cache is not None
    owned.close()
    with ResponseCache() as shared:
        borrower = HFChatModel("fake/model", cache=shared)
        borrower.close()
        shared.put("k", "m", Completion(text="still open"))


def test_generation_works_when_torch_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The autograd guard is an optimisation; without torch there is nothing to guard."""
    monkeypatch.setattr(llm, "_optional_import", _no_optional_imports)
    chat, model, _ = _fake_hf(reply="fine")
    assert chat.complete([_user("hello")], temperature=0.5).text == "fine"
    assert len(model.calls) == 1


def test_loading_without_the_optional_dependencies_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_optional_import", _no_optional_imports)
    chat = HFChatModel("fake/model", device="cpu")
    with pytest.raises(MissingDependencyError, match=r"mcpeval\[hf\]"):
        chat.complete([_user("hello")])


class _StubCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _StubTorch:
    """Just enough torch for device and dtype resolution, seeding and the no-grad guard."""

    def __init__(self, *, cuda: bool = False) -> None:
        self.cuda = _StubCuda(cuda)
        self.float32 = "float32"
        self.float16 = "float16"
        self.bfloat16 = "bfloat16"
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def inference_mode(self) -> AbstractContextManager[None]:
        return nullcontext()


class _StubPretrained:
    """Records how `from_pretrained` was called and hands back a prepared fake."""

    def __init__(self, produced: Any) -> None:
        self.produced = produced
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, name: str, **kwargs: Any) -> Any:
        self.calls.append((name, dict(kwargs)))
        return self.produced


class _LoadableModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.device: str | None = None
        self.eval_calls = 0

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        self.eval_calls += 1


def _stub_libraries(
    monkeypatch: pytest.MonkeyPatch, torch: _StubTorch, model: Any, tokenizer: Any
) -> SimpleNamespace:
    transformers = SimpleNamespace(
        AutoTokenizer=_StubPretrained(tokenizer), AutoModelForCausalLM=_StubPretrained(model)
    )
    modules: dict[str, Any] = {"torch": torch, "transformers": transformers}
    monkeypatch.setattr(llm, "_optional_import", lambda name: modules.get(name))
    return transformers


def test_optional_import_reports_absence_instead_of_raising() -> None:
    assert llm._optional_import("json") is json
    assert llm._optional_import("mcpeval_no_such_package") is None


def test_weights_are_loaded_once_with_the_requested_device_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = _LoadableModel(), _FakeTokenizer()
    transformers = _stub_libraries(monkeypatch, _StubTorch(), model, tokenizer)
    chat = HFChatModel("tiny/model", dtype="float16")
    before = chat.loaded

    chat.complete([_user("hello")], max_tokens=8)
    assert (before, chat.loaded) == (False, True), "the weights load on first use, not before"
    assert transformers.AutoTokenizer.calls == [("tiny/model", {})]
    assert transformers.AutoModelForCausalLM.calls == [("tiny/model", {"dtype": "float16"})]
    assert (model.device, model.eval_calls) == ("cpu", 1)

    chat.complete([_user("again")], max_tokens=8)
    assert len(transformers.AutoModelForCausalLM.calls) == 1, "the weights load once"


def test_sampling_seeds_the_generator_for_reproducibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _StubTorch()
    _stub_libraries(monkeypatch, torch, _LoadableModel(), _FakeTokenizer())
    chat = HFChatModel("tiny/model", seed=1234)
    chat.complete([_user("hello")], max_tokens=8, temperature=0.7)
    chat.complete([_user("hello")], max_tokens=8)
    assert torch.seeds == [1234], "only sampling needs a seed"


def test_dtype_is_chosen_for_the_resolved_device(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = _stub_libraries(
        monkeypatch, _StubTorch(cuda=True), _LoadableModel(), _FakeTokenizer()
    )
    HFChatModel("tiny/model").complete([_user("hello")], max_tokens=8)
    assert transformers.AutoModelForCausalLM.calls[0][1] == {"dtype": "bfloat16"}


@pytest.mark.parametrize(
    ("spec", "torch_available", "cuda", "expected"),
    [
        ("cpu", True, True, "cpu"),
        ("cuda:1", False, False, "cuda:1"),
        ("auto", True, True, "cuda"),
        ("auto", True, False, "cpu"),
        ("auto", False, False, "cpu"),
    ],
)
def test_device_resolution(spec: str, torch_available: bool, cuda: bool, expected: str) -> None:
    torch = _StubTorch(cuda=cuda) if torch_available else None
    assert _resolve_device(torch, spec) == expected


@pytest.mark.parametrize(
    ("spec", "device", "expected"),
    [
        ("auto", "cpu", "float32"),
        ("auto", "cuda:0", "bfloat16"),
        ("float16", "cpu", "float16"),
    ],
)
def test_dtype_resolution(spec: str, device: str, expected: str) -> None:
    assert _resolve_dtype(_StubTorch(), spec, device) == expected


def test_an_unknown_dtype_is_rejected_by_name() -> None:
    with pytest.raises(ChatModelError, match="unknown dtype 'float8'"):
        _resolve_dtype(_StubTorch(), "float8", "cpu")


@pytest.mark.network
@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None,
    reason="the optional hf extra is not installed",
)
def test_a_real_tiny_model_answers_through_the_cache(tmp_path: Path) -> None:
    """End-to-end against real weights; downloads, so it is excluded from CI."""
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        chat = HFChatModel(
            "hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32", cache=cache
        )
        first = chat.complete([_user("Hello")], max_tokens=4)
        assert first.usage.prompt_tokens > 0
        assert chat.complete([_user("Hello")], max_tokens=4) == first
        assert cache.hits == 1
