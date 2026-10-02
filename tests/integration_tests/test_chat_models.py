"""Integration tests against the live FlexAI API.

Require FLEXAI_API_KEY and a funded org; skipped otherwise.
"""

import os

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from langchain_flexai import ChatFlexAI

pytestmark = pytest.mark.skipif(
    not os.environ.get("FLEXAI_API_KEY"),
    reason="FLEXAI_API_KEY not set",
)

# Non-reasoning model, so assertions are about the integration rather than a
# model spending its budget on hidden reasoning.
MODEL = os.environ.get("FLEXAI_TEST_MODEL", "DeepSeek-V4-Flash-0731")
VISION_MODEL = os.environ.get("FLEXAI_TEST_VISION_MODEL", "gemma-4-31b-it")

# A 64x64 solid red PNG (220, 20, 20), generated and byte-verified rather
# than pasted: a truncated fixture still has a valid header, and a model
# describing the broken result reads as a vision failure.
RED_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeUlEQVR4nO3PQQkAMAzAwIqo"
    "f2UTMxF7HINABFzm7H7dcEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCA"
    "FjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFj12qxUDx"
    "eFqrFAAAAABJRU5ErkJggg=="
)


def test_invoke() -> None:
    llm = ChatFlexAI(model=MODEL, max_tokens=64, temperature=0)
    result = llm.invoke("Reply with exactly: ok")
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert result.content.strip()


async def test_ainvoke() -> None:
    llm = ChatFlexAI(model=MODEL, max_tokens=64, temperature=0)
    result = await llm.ainvoke("Reply with exactly: ok")
    assert isinstance(result, AIMessage)
    assert result.content


def test_stream() -> None:
    llm = ChatFlexAI(model=MODEL, max_tokens=64, temperature=0)
    chunks = list(llm.stream("Count from 1 to 5."))
    assert len(chunks) > 1
    assert all(isinstance(c, AIMessageChunk) for c in chunks)


def test_usage_metadata_is_reported() -> None:
    llm = ChatFlexAI(model=MODEL, max_tokens=64, temperature=0)
    result = llm.invoke("Reply with exactly: ok")
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] > 0
    assert result.usage_metadata["output_tokens"] > 0


def test_tool_calling() -> None:
    class GetWeather(BaseModel):
        """Get the current weather in a location."""

        location: str = Field(description="City name")

    llm = ChatFlexAI(model=MODEL, max_tokens=256, temperature=0)
    result = llm.bind_tools([GetWeather]).invoke("What is the weather in Paris?")
    assert isinstance(result, AIMessage)
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "GetWeather"
    assert "paris" in str(result.tool_calls[0]["args"]).lower()


def test_structured_output() -> None:
    class Person(BaseModel):
        """A person."""

        name: str
        age: int

    llm = ChatFlexAI(model=MODEL, max_tokens=256, temperature=0)
    result = llm.with_structured_output(Person).invoke("Invent a person.")
    assert isinstance(result, Person)
    assert result.name
    assert isinstance(result.age, int)


def test_vision() -> None:
    llm = ChatFlexAI(model=VISION_MODEL, max_tokens=512, temperature=0)
    result = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "What colour fills this? One word."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{RED_PNG}"},
                    },
                ]
            )
        ]
    )
    assert "red" in str(result.content).lower()


# A model that returns a reasoning trace in `reasoning_content`. Not all
# reasoning models do -- DeepSeek-V4-Flash reasons inline in `content` -- so
# this is pinned to one that demonstrably emits the field.
REASONING_MODEL = os.environ.get("FLEXAI_TEST_REASONING_MODEL", "gpt-oss-120b")


def test_reasoning_content_surfaced_on_invoke() -> None:
    llm = ChatFlexAI(model=REASONING_MODEL, max_tokens=800, temperature=0)
    result = llm.invoke("What is 17*23? Think it through.")
    assert result.additional_kwargs.get("reasoning_content")


def test_reasoning_content_surfaced_on_stream() -> None:
    llm = ChatFlexAI(model=REASONING_MODEL, max_tokens=800, temperature=0)
    traces = [
        chunk.additional_kwargs.get("reasoning_content")
        for chunk in llm.stream("What is 12*9? Think it through.")
    ]
    assert any(traces), "reasoning_content present on invoke but absent on stream"
