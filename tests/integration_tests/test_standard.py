"""LangChain standard conformance suite (integration, live API)."""

import os
from typing import Any

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_flexai import ChatFlexAI

pytestmark = pytest.mark.skipif(
    not os.environ.get("FLEXAI_API_KEY"), reason="FLEXAI_API_KEY not set"
)


class TestChatFlexAIIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatFlexAI]:
        return ChatFlexAI

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {
            "model": os.environ.get("FLEXAI_TEST_MODEL", "DeepSeek-V4-Flash-0731"),
            "temperature": 0,
        }


class TestChatFlexAIVisionIntegration(ChatModelIntegrationTests):
    """Second pass over a vision model, so image support is exercised.

    The default suite above runs a text-only model, which makes every image
    test skip. A skip is not a pass, and FlexAI serves vision models, so this
    class declares image support and runs the same suite against one.
    """

    @property
    def chat_model_class(self) -> type[ChatFlexAI]:
        return ChatFlexAI

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {
            "model": os.environ.get("FLEXAI_TEST_VISION_MODEL", "gemma-4-31b-it"),
            "temperature": 0,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        # Forced tool choice is per-model on FlexAI, not provider-wide. It is
        # best-effort rather than constrained decoding: the gateway returns
        # 400 tool_choice_not_honored when the model declines. gemma-4-31b-it
        # declines on a prompt that invites no tool call; DeepSeek-V4-Flash
        # and gpt-oss-120b honour it, which is why the suite above leaves this
        # at its default.
        return False
