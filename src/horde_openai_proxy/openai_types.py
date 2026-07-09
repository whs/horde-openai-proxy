from typing import Optional, List, Union, Literal

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """An OpenAI Chat Completion request."""

    model_config = {"extra": "ignore"}

    messages: list[dict] = Field(min_length=1)
    model: str
    tools: Optional[list[dict]] = Field(None)
    frequency_penalty: Optional[float] = Field(None)
    presence_penalty: Optional[float] = Field(None)
    max_tokens: int = Field(512)
    n: Optional[int] = Field(1)
    stop: List[str] = Field(default_factory=list)
    temperature: Optional[float] = Field(None)
    top_p: Optional[float] = Field(None)
    timeout: int = Field(300)

    # Custom params
    trusted_workers: bool = Field(False)
    validated_backends: bool = Field(False)
    slow_workers: bool = Field(True)
    allow_downgrade: bool = Field(False)


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
