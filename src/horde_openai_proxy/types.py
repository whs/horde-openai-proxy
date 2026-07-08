from typing import Any, Optional, Union, List

from pydantic import BaseModel, Field, RootModel


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


class ModelGenerationInput(BaseModel):
    """A (partial) KoboldAI generation input."""

    model_config = {"extra": "ignore"}

    max_context_length: int = 2048
    max_length: Optional[int] = 512
    n: Optional[int] = 1
    rep_pen: Optional[float] = None
    stop_sequence: List[str] = []
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    # Additional formatting options, handled by the proxy. It uses KoboldAI's GUI defaults!
    frmtadsnsp: bool = True
    frmtrmblln: bool = False
    frmtrmspch: bool = False
    frmttriminc: bool = True
    singleline: bool = False


class HordeRequest(BaseModel):
    """A request to the Horde API."""

    prompt: str = Field(...)
    models: List[str] = Field(...)
    timeout: int = Field((60 * 20) - 30)
    params: ModelGenerationInput


class TextGeneration(BaseModel):
    """A generated text returned from the Horde."""

    uuid: str
    model: str
    text: str
    kudos: int


class HordeModelInfo(BaseModel):
    """Schema of https://github.com/Haidra-Org/AI-Horde-text-model-reference/blob/main/db.json"""

    name: str
    model_name: str
    display_name: str
    description: str
    baseline: str
    parameters: Optional[float] = None
    version: str
    style: str
    nsfw: bool
    url: Optional[str] = None
    instruct_format: Optional[str] = None
    tags: list[str] = []
    settings: dict[str, Any] = {}


HordeModelInfoResponse = RootModel[dict[str, HordeModelInfo]]
