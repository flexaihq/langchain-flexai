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

Models that emit a reasoning trace return it in
`additional_kwargs["reasoning_content"]` alongside the answer.

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
