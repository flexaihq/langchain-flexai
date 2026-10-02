"""Unit tests for ChatFlexAI. No network."""

import os
from unittest import mock

import pytest
from pydantic import SecretStr

from langchain_flexai import ChatFlexAI, __version__
from langchain_flexai.chat_models import DEFAULT_API_BASE

MODEL = "test-vendor/test-model"


def test_version_matches_package() -> None:
    assert __version__


def test_defaults_to_flexai_endpoint() -> None:
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("k"))
    assert llm.api_base == DEFAULT_API_BASE
    assert llm.root_client.base_url.host == "api.flex.ai"


def test_api_key_read_from_environment() -> None:
    with mock.patch.dict(os.environ, {"FLEXAI_API_KEY": "from-env"}, clear=True):
        llm = ChatFlexAI(model=MODEL)
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "from-env"


def test_explicit_api_key_wins_over_environment() -> None:
    with mock.patch.dict(os.environ, {"FLEXAI_API_KEY": "from-env"}, clear=True):
        llm = ChatFlexAI(model=MODEL, api_key=SecretStr("explicit"))
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "explicit"


def test_base_url_overridable_by_environment() -> None:
    with mock.patch.dict(
        os.environ,
        {"FLEXAI_API_KEY": "k", "FLEXAI_API_BASE": "https://eu.example/v1"},
        clear=True,
    ):
        llm = ChatFlexAI(model=MODEL)
    assert llm.api_base == "https://eu.example/v1"


def test_base_url_alias_accepted() -> None:
    llm = ChatFlexAI(
        model=MODEL, api_key=SecretStr("k"), base_url="https://other.example/v1"
    )
    assert llm.api_base == "https://other.example/v1"


def test_api_key_is_not_printed() -> None:
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("super-secret"))
    assert "super-secret" not in repr(llm)
    assert "super-secret" not in str(llm)


def test_lc_secrets_names_the_env_var() -> None:
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("k"))
    assert llm.lc_secrets == {"api_key": "FLEXAI_API_KEY"}


def test_llm_type_and_namespace() -> None:
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("k"))
    assert llm._llm_type == "chat-flexai"
    assert ChatFlexAI.get_lc_namespace() == ["langchain", "chat_models", "flexai"]


def test_async_client_targets_the_same_endpoint() -> None:
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("k"))
    assert llm.root_async_client.base_url.host == "api.flex.ai"


def test_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="n must be at least 1"):
        ChatFlexAI(model=MODEL, api_key=SecretStr("k"), n=0)


def test_n_greater_than_one_rejected_when_streaming() -> None:
    with pytest.raises(ValueError, match="n must be 1 when streaming"):
        ChatFlexAI(model=MODEL, api_key=SecretStr("k"), n=2, streaming=True)


def test_missing_key_fails_fast_naming_the_flexai_variable() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="FLEXAI_API_KEY must be set"):
            ChatFlexAI(model=MODEL)


def test_missing_key_also_fails_against_a_custom_endpoint() -> None:
    # One error for one mistake: without this the OpenAI client raises its own,
    # naming OPENAI_API_KEY, which sends the reader to the wrong variable.
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="FLEXAI_API_KEY must be set"):
            ChatFlexAI(model=MODEL, base_url="http://localhost:8000/v1")


def test_stream_usage_defaults_on() -> None:
    # FlexAI only emits usage on a stream when stream_options.include_usage is
    # sent, so the ChatOpenAI default of None yields usage_metadata=None for
    # every streamed response.
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("k"))
    assert llm.stream_usage is True


def test_stream_usage_can_be_disabled() -> None:
    llm = ChatFlexAI(model=MODEL, api_key=SecretStr("k"), stream_usage=False)
    assert llm.stream_usage is False
