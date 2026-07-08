import logging
from typing import List

from fastapi import FastAPI, HTTPException, Request

from horde_openai_proxy import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Model,
    openai_to_horde_async,
    completions_to_openai_response,
    get_horde_completion_async,
    completions_to_openai_response_async,
)

from horde_openai_proxy.model import get_models_async

app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/v1/chat/models")
async def get_chat_models() -> List[Model]:
    # TODO: Check openai spec
    return list((await get_models_async()).values())


@app.post("/v1/chat/completions")
async def post_chat_completion(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse:
    token = (
        request.headers.get("authorization", "0000000000")
        .lstrip("Bearer ")
        .lstrip("sk-")
    )

    try:
        horde_request = await openai_to_horde_async(body)
        completions = await get_horde_completion_async(
            token,
            horde_request,
            trusted_workers=body.trusted_workers,
            validated_backends=body.validated_backends,
            slow_workers=body.slow_workers,
            allow_downgrade=body.allow_downgrade,
        )
    except ValueError as e:
        logging.warning("Request error", exc_info=e)
        raise HTTPException(status_code=406, detail=str(e))

    return await completions_to_openai_response_async(completions, horde_request.prompt)
