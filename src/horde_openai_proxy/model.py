import re
import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
from cachetools import TTLCache
from cachetools_async import cached as cached_async

from .horde import get_horde_models_async
from .types import HordeModelInfoResponse, HordeModelInfo

KNOWN_ENGINES = {"aphrodite", "koboldcpp"}
QUANTS_RE = re.compile(r"(-)*(([i]*q[1-9](_[k0])*(_[x]*[sml])*)|f[48])(-)*", re.I)


def estimate_hf_url(
    name: str, references: Optional[HordeModelInfoResponse] = None
) -> str:
    if references is not None and name in references.root:
        url = references.root[name].url
        if url is not None and url.startswith("https://huggingface.co/"):
            return url

    name_no_quant = QUANTS_RE.sub("", name, 1)
    if name_no_quant != name:
        return estimate_hf_url(name_no_quant, references)
    if name_no_quant.count("_") == 1:
        # kobold likes to replace / with _, but the quant has _
        return estimate_hf_url(name_no_quant.replace("_", "/"))

    parts = name_no_quant.split("/")
    if len(parts) == 3 or parts[0] in KNOWN_ENGINES:
        # If the name is engine/user/model_name then try this algorithm again with just user/model_name
        return estimate_hf_url("/".join(parts[1:]), references)

    if len(parts) == 2:
        return f"https://huggingface.co/{parts[0]}/{parts[1]}"

    return None

@dataclass
class Model:
    name: str
    hf_url: str
    known_to_horde: bool
    reference: Optional[HordeModelInfo]


def get_references() -> HordeModelInfoResponse:
    """The references are known models, with usually more accurate information than the guesses."""
    return asyncio.run(get_references_async())


@cached_async(TTLCache(maxsize=1, ttl=86400))
async def get_references_async() -> HordeModelInfoResponse:
    """The references are known models, with usually more accurate information than the guesses."""
    async with httpx.AsyncClient() as client:
        return HordeModelInfoResponse.parse_obj(
            (
                await client.get(
                    "https://raw.githubusercontent.com/db0/AI-Horde-text-model-reference/main/db.json"
                )
            ).json()
        )


def get_models() -> dict[str, Model]:
    """
    Get all models from the Horde API, with estimated sizes, base models, templates, etc.
    """
    return asyncio.run(get_models_async())


@cached_async(TTLCache(maxsize=1, ttl=3600))
async def get_models_async() -> dict[str, Model]:
    """
    Get all models from the Horde API, with estimated sizes, base models, templates, etc.
    """
    references, server_models = await asyncio.gather(
        get_references_async(), get_horde_models_async()
    )

    models = {}
    for model in server_models:
        name = model["name"]
        if "/" in name:
            hf_url = estimate_hf_url(name, references)

            if hf_url is None:
                print(f"Unknown model: {name}, ignoring.")
            else:
                models[name] = Model(
                    name=name,
                    hf_url=hf_url,
                    known_to_horde=name in references.root,
                    reference=references.root.get(name),
                )
    return models


async def get_model(name: str) -> Optional[Model]:
    models = await get_models_async()
    return models.get(name, None)
