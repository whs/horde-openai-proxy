from typing import Any, Optional, List, Literal, Union

from pydantic import BaseModel, Field, RootModel


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
    min_p: Optional[float] = None
    top_a: Optional[float] = None
    top_k: Optional[int] = None

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


class HordeWorkerInfo(BaseModel):
    model_config = {"extra": "ignore"}

    type: Union[Literal["text"], Literal["image"], Literal["interrogation"]]
    name: str
    id: str
    online: bool
    requests_fulfilled: int
    kudos_rewards: int
    performance: str
    threads: int
    uptime: float
    maintenance_mode: bool
    info: Optional[str] = None
    nsfw: bool
    owner: Optional[str] = None
    trusted: bool
    flagged: bool
    uncompleted_jobs: int
    models: List[str]
    forms: Optional[List[str]] = None
    bridge_agent: str
    max_pixels: Optional[int] = None
    megapixelsteps_generated: Optional[float] = None
    img2img: Optional[bool] = None
    painting: Optional[bool] = None
    post_processing: Optional[bool] = Field(None, alias="post-processing")
    lora: Optional[bool] = None
    controlnet: Optional[bool] = None
    sdxl_controlnet: Optional[bool] = None
    max_length: int
    max_context_length: int
    tokens_generated: Optional[int] = None


HordeWorkerListResponse = RootModel[list[HordeWorkerInfo]]
