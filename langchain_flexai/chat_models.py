"""FlexAI chat model integration."""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://api.flex.ai/v1"


class ChatFlexAI(BaseChatOpenAI):
    """FlexAI chat model.

    FlexAI serves open-weight models behind an OpenAI-compatible API, so this
    class is a thin configuration of LangChain's OpenAI client: it supplies the
    FlexAI base URL and reads ``FLEXAI_API_KEY`` from the environment. Anything
    ``ChatOpenAI`` supports is supported here.

    Setup:
        Install ``langchain-flexai`` and set the environment variable
        ``FLEXAI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-flexai
            export FLEXAI_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of a FlexAI-served model, as returned by ``GET /v1/models``
            (for example ``"DeepSeek-V4-Flash-0731"``). FlexAI requires the
            canonical id -- the bare model name, without the organisation
            prefix.
        temperature: float | None
            Sampling temperature.
        max_tokens: int | None
            Maximum number of tokens to generate.

    Key init args — client params:
        api_key: SecretStr | None
            FlexAI API key. Defaults to ``FLEXAI_API_KEY``.
        api_base: str
            Endpoint URL. Defaults to ``FLEXAI_API_BASE``, else
            ``https://api.flex.ai/v1``.
        timeout: float | tuple[float, float] | Any | None
            Request timeout.
        max_retries: int
            Number of retries on a failed request.

    Instantiate:
        .. code-block:: python

            from langchain_flexai import ChatFlexAI

            llm = ChatFlexAI(
                model="DeepSeek-V4-Flash-0731",
                temperature=0,
                max_tokens=4096,
            )

    Invoke:
        .. code-block:: python

            llm.invoke("Explain speculative decoding in two sentences.")

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location.'''

                location: str = Field(description="City, e.g. Paris")


            llm.bind_tools([GetWeather]).invoke("Weather in Paris?")

    Structured output:
        .. code-block:: python

            llm.with_structured_output(GetWeather).invoke("Weather in Paris?")

        Not every served model enforces a strict JSON schema; the models that
        do not will reject the request rather than silently ignore it. Check
        ``GET /v1/models`` for a model's supported parameters.

    Reasoning models:
        Models that emit a reasoning trace return it in
        ``additional_kwargs["reasoning_content"]`` alongside the answer.

    """

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(alias="model")
    """The name of the FlexAI model to use."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("FLEXAI_API_KEY", default=None),
    )
    """FlexAI API key."""

    stream_usage: bool = True
    """Whether to include token usage in streaming output.

    Defaults to ``True``, unlike ``ChatOpenAI``. FlexAI only emits usage on a
    stream when ``stream_options.include_usage`` is sent, so leaving this unset
    makes ``usage_metadata`` ``None`` for every streamed response and silently
    breaks cost tracking. Set it to ``False`` to opt out.
    """

    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("FLEXAI_API_BASE", default=DEFAULT_API_BASE),
    )
    """FlexAI API base URL."""

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map constructor args to the environment variables holding them."""
        return {"api_key": "FLEXAI_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "chat-flexai"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Return the namespace of this object."""
        return ["langchain", "chat_models", "flexai"]

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Build the sync and async clients against the FlexAI endpoint."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        if not (self.api_key and self.api_key.get_secret_value()):
            # Unconditional, including for a custom api_base: the underlying
            # OpenAI client refuses to construct without credentials anyway,
            # and raising here gives one clear error naming the FlexAI variable
            # instead of an OpenAI one that names OPENAI_API_KEY.
            msg = "FLEXAI_API_KEY must be set, or api_key passed explicitly."
            raise ValueError(msg)

        # Drop unset values rather than forwarding them: the OpenAI client
        # rejects an explicit None for max_retries, and LangChain leaves these
        # unset by default.
        client_params: dict[str, Any] = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Surface ``reasoning_content``, which the OpenAI client drops.

        Several FlexAI-served models return their reasoning trace in a
        ``reasoning_content`` field alongside ``content``. It is not part of
        the OpenAI schema, so the SDK parses it into ``model_extra`` and
        ``BaseChatOpenAI`` discards it. Lift it onto the message instead of
        silently losing it.
        """
        result = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return result

        choices = getattr(response, "choices", None)
        if not choices or not result.generations:
            return result

        message = choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                reasoning = extra.get("reasoning_content")
        if reasoning:
            result.generations[0].message.additional_kwargs["reasoning_content"] = (
                reasoning
            )
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Carry ``reasoning_content`` through streaming deltas too.

        Without this the trace is available on an ``invoke`` and missing on a
        ``stream`` of the same model, which is a worse contract than not
        exposing it at all.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices") or []
        if not choices:
            return generation_chunk
        reasoning = (choices[0].get("delta") or {}).get("reasoning_content")
        if reasoning:
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk
