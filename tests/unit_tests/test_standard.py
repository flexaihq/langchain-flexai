"""LangChain standard conformance suite (unit)."""

from typing import Any

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_flexai import ChatFlexAI


class TestChatFlexAIUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatFlexAI]:
        return ChatFlexAI

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": "test-vendor/test-model", "api_key": "test-key"}
