# langchain-flexai

LangChain integration for [FlexAI](https://flex.ai) — open-weight models served
behind an OpenAI-compatible API.

## Installation

```bash
pip install -U langchain-flexai
export FLEXAI_API_KEY="your-api-key"
```

## Chat models

```python
from langchain_flexai import ChatFlexAI

llm = ChatFlexAI(model="DeepSeek-V4-Flash-0731")
llm.invoke("Explain speculative decoding in two sentences.")
```

`model` takes the canonical id as returned by `GET /v1/models` — the bare model
name, without the organisation prefix.

### Tool calling

```python
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(description="City, e.g. Paris")


llm.bind_tools([GetWeather]).invoke("Weather in Paris?")
```

### Structured output

```python
llm.with_structured_output(GetWeather).invoke("Weather in Paris?")
```

Not every served model enforces a strict JSON schema. Models that do not will
reject the request rather than silently ignore it — check a model's
`supported_parameters` in `GET /v1/models`.

### Reasoning models

Some FlexAI-served models return a reasoning trace in a `reasoning_content`
field. That field is not part of the OpenAI schema, so `ChatOpenAI` discards
it; this package surfaces it on both `invoke` and `stream`:

```python
result = ChatFlexAI(model="gpt-oss-120b").invoke("What is 17*23?")
result.additional_kwargs["reasoning_content"]
```

Not every reasoning model uses the field — some reason inline in `content`
instead — so treat it as present-or-absent rather than guaranteed.

## Capabilities vary by model

FlexAI serves many models through one endpoint, so some behaviour is per-model
rather than provider-wide. Verified against the live API:

| Behaviour | Notes |
|---|---|
| Streaming token usage | Works. This package sets `stream_usage=True` by default, unlike `ChatOpenAI`, because FlexAI only reports usage when `stream_options.include_usage` is sent. |
| Forced tool choice | Per-model, and best-effort rather than constrained decoding. `DeepSeek-V4-Flash-0731` and `gpt-oss-120b` honour `tool_choice="any"` on a prompt that invites no tool call; `gemma-4-31b-it` declines, and the gateway returns `400 tool_choice_not_honored` rather than forcing one. |
| Structured output | Strict JSON schema is enforced on a subset of models. Those that do not support it reject the request rather than silently ignoring it. |
| Image input | Supported on vision models only. A model that is not a vision model will not read the image. |

Check a model's `supported_parameters` in `GET /v1/models` before relying on
any of these.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FLEXAI_API_KEY` | — | API key. Required. |
| `FLEXAI_API_BASE` | `https://api.flex.ai/v1` | Endpoint, for regional or self-hosted deployments. |

## Relationship to `langchain-openai`

FlexAI is OpenAI-compatible, so this package is a thin configuration of
`BaseChatOpenAI` rather than a separate client. Everything `ChatOpenAI`
supports works here. Using `ChatOpenAI` with `base_url="https://api.flex.ai/v1"`
remains equivalent and supported; this package just supplies the defaults.

## Documentation

[docs.flex.ai/inference-api/agents/langchain](https://docs.flex.ai/inference-api/agents/langchain)

## License

MIT
