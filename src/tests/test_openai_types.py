import pytest
from pydantic import ValidationError

from horde_openai_proxy.openai_types import (
    ChatCompletionAssistantMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionReasoningDetails,
    ChatCompletionRequest,
    ChatCompletionUserMessage,
    parse_json,
    ToolCallFunction,
    ToolCallFunctionResponse,
    ChatCompletionMessageTextContent,
    ChatCompletionStreamingChunkChoiceDelta,
)

TEST_MODEL = "google/gemma-4-E2B-it"

@pytest.mark.parametrize(
    "value",
    [
        {"model": TEST_MODEL, "messages": [ChatCompletionUserMessage(content="hi")], "response_format": {"type": "json_object"}},
        {"messages": [ChatCompletionUserMessage(content="hi")]},
        {"model": TEST_MODEL, "messages": []},
    ]
)
def test_chatcompletionrequest_error(value):
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(value)

def test_chatcompletionrequest_minimal():
    req = ChatCompletionRequest(model=TEST_MODEL, messages=[ChatCompletionUserMessage(content="hi")])
    assert req.max_tokens == 512
    assert req.max_completion_tokens == 512
    assert req.n == 1
    assert req.timeout == 300

def test_chatcompletionrequest_max_tokens():
    req = ChatCompletionRequest(model=TEST_MODEL, messages=[ChatCompletionUserMessage(content="hi")], max_tokens=100)
    assert req.max_tokens == 100
    assert req.max_completion_tokens == 100


def test_chatcompletionrequest_max_completion_tokens():
    req = ChatCompletionRequest(model=TEST_MODEL, messages=[ChatCompletionUserMessage(content="hi")], max_completion_tokens=100)
    assert req.max_tokens == 100
    assert req.max_completion_tokens == 100


def test_chatcompletionrequest_model_single():
    req = ChatCompletionRequest(model=TEST_MODEL, messages=[ChatCompletionUserMessage(content="hi")])
    assert req.model == TEST_MODEL
    assert req.models == [TEST_MODEL]


def test_chatcompletionrequest_model_comma():
    req = ChatCompletionRequest(model=f"{TEST_MODEL},b", messages=[ChatCompletionUserMessage(content="hi")])
    assert req.model == f"{TEST_MODEL},b"
    assert req.models == [TEST_MODEL, "b"]


def test_chatcompletionrequest_models():
    req = ChatCompletionRequest(models=[TEST_MODEL, "b"], messages=[ChatCompletionUserMessage(content="hi")])
    assert req.model == f"{TEST_MODEL},b"
    assert req.models == [TEST_MODEL, "b"]


def test_toolcallfunction_arguments_dict():
    tool_call = ToolCallFunction(name="weather", arguments={"city": "London"})
    assert tool_call.arguments == {"city": "London"}


def test_toolcallfunction_arguments_json():
    tool_call = ToolCallFunction(name="weather", arguments='{"city": "London"}')
    assert tool_call.arguments == {"city": "London"}


def test_chatcompletionassistantmessage_thinking():
    msg = ChatCompletionAssistantMessage(
        content="Hello!",
        role="assistant",
        thinking="think",
    )
    assert msg.thinking == "think"
    assert msg.reasoning_details == [ChatCompletionReasoningDetails(text="think")]


def test_chatcompletionassistantmessage_reasoning_details():
    msg = ChatCompletionAssistantMessage(
        content="Hello!",
        role="assistant",
        reasoning_details=[ChatCompletionReasoningDetails(text="think")],
    )
    assert msg.thinking == "think"
    assert msg.reasoning_details == [ChatCompletionReasoningDetails(text="think")]

def test_toolcallfunctionresponse_arguments_dict():
    tool_call = ToolCallFunctionResponse(name="weather", arguments={"city": "London"})
    assert tool_call.arguments == '{"city": "London"}'


def test_toolcallfunctionresponse_arguments_json():
    tool_call = ToolCallFunctionResponse(name="weather", arguments='{"city": "London"}')
    assert tool_call.arguments == '{"city": "London"}'

def test_chatcompletionstreamingchunkchoicedelta_content():
    msg = ChatCompletionStreamingChunkChoiceDelta(role="system", content=ChatCompletionMessageTextContent(text="hi"))
    assert msg.content == "hi"

def test_chatcompletionstreamingchunkchoicedelta_thinking():
    msg = ChatCompletionStreamingChunkChoiceDelta(
        role="assistant",
        thinking="think"
    )
    assert msg.thinking == "think"
    assert msg.reasoning_details == [ChatCompletionReasoningDetails(text="think")]


def test_chatcompletionstreamingchunkchoicedelta_reasoning_details():
    msg = ChatCompletionStreamingChunkChoiceDelta(
        role="assistant",
        reasoning_details=[ChatCompletionReasoningDetails(text="think")],
    )
    assert msg.thinking == "think"
    assert msg.reasoning_details == [ChatCompletionReasoningDetails(text="think")]