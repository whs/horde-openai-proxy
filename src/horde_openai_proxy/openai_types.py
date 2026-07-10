import json
from typing import Optional, List, Union, Literal, Self, Annotated, Any

from pydantic import BaseModel, Field, model_validator, BeforeValidator
from pydantic_core import MISSING


def ensure_str_or_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, bytes):
        return value.decode("utf8")

    try:
        return json.dumps(value)
    except TypeError:
        raise ValueError("value is not serializable")


def parse_json(value: Any) -> dict:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("value is not json")
    elif isinstance(value, BaseModel):
        return value.model_dump()
    elif isinstance(value, dict):
        return value

    raise ValueError("Invalid type")


class ChatCompletionRequest(BaseModel):
    """An OpenAI Chat Completion request."""

    messages: list["ChatCompletionAllMessages"] = Field(min_length=1)
    model: Optional[str] = None
    models: Optional[Annotated[list[str], Field(min_length=1)]] = None
    tools: Optional[list["ToolDefinition"]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[Annotated[int, Field(gt=1)]] = Field(None, deprecated=True)
    max_completion_tokens: Optional[Annotated[int, Field(gt=1)]] = 512
    n: Optional[int] = Field(1)
    stop: List[str] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_a: Optional[float] = None
    top_k: Optional[int] = None
    timeout: int = Field(300)

    # Unsupported params
    response_format: MISSING = MISSING
    stream: Optional[Literal[False]] = False
    tool_choice: Optional[Literal["auto"]] = "auto"

    # Custom params
    trusted_workers: bool = Field(False)
    validated_backends: bool = Field(False)
    slow_workers: bool = Field(True)
    allow_downgrade: bool = Field(False)

    @model_validator(mode="after")
    def check_max_tokens(self) -> Self:
        if self.max_tokens is not None:
            self.max_tokens = self.max_tokens
            self.max_completion_tokens = self.max_tokens
            return self
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
            self.max_completion_tokens = self.max_completion_tokens
            return self

        raise ValueError("max_tokens or max_completion_tokens must be set")

    @model_validator(mode="after")
    def check_models(self) -> Self:
        if self.models is not None:
            self.model = ",".join(self.models)
            return self
        if self.model is not None:
            self.models = self.model.split(",")
            return self
        raise ValueError("one of model or models must be set")


class ChatCompletionMessage(BaseModel):
    content: Union[str, list[str], list["ChatCompletionMessageTextContent"]]
    role: str
    name: Optional[str] = None


class ChatCompletionUserMessage(ChatCompletionMessage):
    role: Literal["user"] = "user"


class ChatCompletionSystemMessage(ChatCompletionMessage):
    role: Literal["system"] = "system"
    content: Union[str, "ChatCompletionMessageTextContent"]


class ChatCompletionAssistantMessage(ChatCompletionMessage):
    role: Literal["assistant"] = "assistant"
    reasoning: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None


class ChatCompletionToolMessage(ChatCompletionMessage):
    content: Union[str, list[str], list["ChatCompletionMessageTextContent"]]
    role: Literal["tool"] = "tool"
    tool_call_id: str


ChatCompletionAllMessages = Union[
    ChatCompletionUserMessage,
    ChatCompletionSystemMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
]


class ChatCompletionMessageTextContent(BaseModel):
    text: str
    type: Literal["text"]


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: "ToolCallFunction"


class ToolCallFunction(BaseModel):
    name: str
    arguments: Annotated[dict, BeforeValidator(parse_json)]


class ToolDefinitionFunction(BaseModel):
    type: Literal["function"] = "function"
    function: "ToolDefinitionFunctionDetail"


class ToolDefinitionFunctionDetail(BaseModel):
    name: str
    description: str = ""
    parameters: dict
    strict: Optional[bool] = None


ToolDefinition = Union[ToolDefinitionFunction]


class ChatCompletionResponse(BaseModel):
    """An OpenAI Chat Completion response."""

    id: str
    choices: list[dict]
    created: int
    model: str
    usage: dict


class ModelResponse(BaseModel):
    # https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties
    data: list["OpenAIModel"]


class OpenAIModel(BaseModel):
    architecture: "ModelArchitecture"
    canonical_slug: str
    context_length: int
    id: str
    name: str
    per_request_limits: "ModelLimit"
    supported_parameters: list[str]
    top_provider: "ModelTopProvider"
    description: str
    hugging_face_id: Optional[str] = None
    pricing: "ModelPricing"


class ModelArchitecture(BaseModel):
    input_modalities: list[Union[Literal["text"], Literal["image"]]]
    modality: str
    output_modalities: list[Union[Literal["text"], Literal["image"]]]


class ModelLimit(BaseModel):
    completion_tokens: int
    prompt_tokens: int


class ModelTopProvider(BaseModel):
    name: str
    context_length: int
    max_completion_tokens: int


class ModelPricing(BaseModel):
    completion: str
    prompt: str
    image: Optional[str] = None
    image_output: Optional[str] = None
    image_token: Optional[str] = None
    input_cache_read: Optional[str] = None
    input_cache_write: Optional[str] = None
    request: Optional[str] = None
