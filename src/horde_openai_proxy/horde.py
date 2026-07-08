import asyncio
import time
from json import JSONDecodeError
from typing import List

import httpx

from .types import HordeRequest, TextGeneration

HORDE_HOST = "https://stablehorde.net/api/"


def remove_stop_words(text: str, stop_sequence: List[str]) -> str:
    """
    Clean up the response text by removing trailing stop words.
    :param text: The text to clean up.
    :param stop_sequence: The stop sequence to remove.
    :return:
    """
    while True:
        removed = False
        for stop_word in stop_sequence:
            text = text.removesuffix(stop_word)
            removed = True

        if not removed:
            return text


def get_data(response: httpx.Response):
    if response.status_code != 200 and response.status_code != 202:
        try:
            message = response.json().get("message")
        except (JSONDecodeError, KeyError):
            message = response.status_code
        raise ValueError(f"Error: {message}")
    return response.json()


def get_horde_completion(*args, **kwargs) -> List[TextGeneration]:
    return asyncio.run(get_horde_completion_async(*args, **kwargs))


async def get_horde_completion_async(
    apikey: str,
    request: HordeRequest,
    *,
    trusted_workers: bool = False,
    validated_backends: bool = False,
    slow_workers: bool = True,
    allow_downgrade: bool = False,
) -> List[TextGeneration]:
    """
    Request text completions from the StableHorde API and awaits the completions.
    Raises a ValueError if the request is not possible, faulted, timed out, or if there are not enough generations.
    :param apikey: API key for the StableHorde API.
    :param request: HordeRequest
    :param trusted_workers: Only use workers that have been trusted.
    :param validated_backends: Only use backends that have been validated.
    :param slow_workers: Allow slow workers to be used.
    :param allow_downgrade: Allow downgrading context length if necessary.
    :return: List of TextGeneration
    :raises ValueError
    """
    async with httpx.AsyncClient(base_url=HORDE_HOST) as client:
        initial_request = get_data(
            await client.post(
                "v2/generate/text/async",
                headers={
                    "apikey": apikey,
                },
                json={
                    "prompt": request.prompt,
                    "models": request.models,
                    "params": request.params.model_dump(exclude_none=True),
                    "trusted_workers": trusted_workers,
                    "validated_backends": validated_backends,
                    "slow_workers": slow_workers,
                    "allow_downgrade": allow_downgrade,
                },
            )
        )

        uuid = initial_request["id"]

        # Await the completion
        initial_time = time.time()
        while time.time() - initial_time < request.timeout:
            data = get_data(await client.get(f"v2/generate/text/status/{uuid}"))

            if not data["is_possible"]:
                raise ValueError("Request is not possible.")

            if data["faulted"]:
                raise ValueError("Request errored.")

            if not data["done"]:
                await asyncio.sleep(0.5)
                continue

            if len(data["generations"]) < request.params.n:
                raise ValueError("Not enough generations.")

            # Parse the generations
            generations = []
            for generation in data["generations"]:
                generations.append(
                    TextGeneration(
                        uuid=str(uuid),
                        model=generation["model"],
                        text=generation["text"],
                        kudos=data["kudos"],
                    )
                )
            return generations

        raise ValueError("Request timed out.")


def get_horde_models() -> List[dict]:
    """
    Get the models available on the StableHorde API.
    :return: List of models.
    :raises ValueError
    """
    return asyncio.run(get_horde_models_async())


async def get_horde_models_async() -> List[dict]:
    """
    Get the models available on the StableHorde API.
    :return: List of models.
    :raises ValueError
    """
    async with httpx.AsyncClient(base_url=HORDE_HOST) as client:
        return get_data(
            await client.get(
                "v2/status/models",
                params={
                    "type": "text",
                    "min_count": 1,
                },
            )
        )
