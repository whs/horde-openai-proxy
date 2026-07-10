import asyncio
import logging
import time
from typing import List, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request

# fastapi.sse is not used because it cannot be used outside response_class
from sse_starlette import EventSourceResponse, ServerSentEvent

from horde_openai_proxy import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Model,
    openai_to_horde_async,
    get_horde_completion_async,
    completions_to_openai_response_async,
)
from horde_openai_proxy.horde import get_horde_workers
from horde_openai_proxy.model import (
    get_models_async,
    get_references_async,
    estimate_hf_url,
)
from horde_openai_proxy.openai_types import (
    ModelArchitecture,
    ModelLimit,
    ModelPricing,
    ModelResponse,
    ModelTopProvider,
    OpenAIModel,
    ChatCompletionStreamingChunk,
    ChatCompletionStreamingChunkChoice,
    FinishReason,
    ChatCompletionStreamingChunkChoiceDelta,
    ErrorMessage,
)

app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/v1/models")
async def get_models() -> ModelResponse:
    workers, models_ref = await asyncio.gather(
        get_horde_workers(), get_references_async()
    )
    out: dict[str, OpenAIModel] = {}

    for worker in workers.root:
        my_provider = ModelTopProvider(
            name=worker.name,
            context_length=max(0, worker.max_context_length - worker.max_length),
            max_completion_tokens=worker.max_length,
        )
        for model in worker.models:
            hf_id = estimate_hf_url(model, models_ref)

            if hf_id is not None:
                hf_id = hf_id.removeprefix("https://huggingface.co/")

            model_info = models_ref.root.get(model, None)

            prev_model = out.get(model, None)
            best_provider = my_provider

            if prev_model:
                if prev_model.top_provider.context_length >= my_provider.context_length:
                    best_provider = prev_model.top_provider

            supported_parameters = [
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "n",
                "stop",
                "temperature",
                "top_p",
            ]

            if hf_id:
                supported_parameters.append("tools")

            out[model] = OpenAIModel(
                architecture=ModelArchitecture(
                    input_modalities=["text"],
                    modality="text->text",
                    output_modalities=["text"],
                ),
                canonical_slug=model,
                context_length=best_provider.context_length,
                id=model,
                name=model_info.name if model_info is not None else model,
                per_request_limits=ModelLimit(
                    completion_tokens=best_provider.max_completion_tokens,
                    prompt_tokens=best_provider.context_length,
                ),
                supported_parameters=supported_parameters,
                top_provider=best_provider,
                description=model_info.description if model_info is not None else "",
                hugging_face_id=hf_id,
                pricing=ModelPricing(
                    completion="0",
                    prompt="0",
                    input_cache_read="0",
                    input_cache_write="0",
                    request="0",
                ),
            )

    return ModelResponse(data=list(out.values()))


@app.get("/v1/chat/models")
async def get_chat_models() -> List[Model]:
    return list((await get_models_async()).values())


@app.post("/v1/chat/completions", response_model=None)
async def post_chat_completion(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse | EventSourceResponse:
    token = (
        request.headers.get("authorization", "0000000000")
        .lstrip("Bearer ")
        .lstrip("sk-")
    )

    if body.stream:
        return EventSourceResponse(
            _post_chat_completion_streaming(request, body, token)
        )
    else:
        return await _post_chat_completion(request, body, token)


async def _post_chat_completion(
    request: Request, body: ChatCompletionRequest, token: str
) -> ChatCompletionResponse:
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


async def _post_chat_completion_streaming(
    request: Request, body: ChatCompletionRequest, token: str
) -> AsyncGenerator[ServerSentEvent]:
    try:
        horde_request = await openai_to_horde_async(body)

        # TODO: Use progress
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
        yield ServerSentEvent(
            ChatCompletionStreamingChunk(
                id="",
                choices=[
                    ChatCompletionStreamingChunkChoice(
                        index=0,
                        finish_reason=FinishReason.Error,
                        delta=ChatCompletionStreamingChunkChoiceDelta(role="system"),
                    )
                ],
                created=int(time.time()),
                model=body.model,
                error=ErrorMessage(message=str(e)),
            ).model_dump_json()
        )
        return

    out = await completions_to_openai_response_async(completions, horde_request.prompt)
    yield ServerSentEvent(
        ChatCompletionStreamingChunk(
            id=out.id,
            choices=[
                ChatCompletionStreamingChunkChoice.from_choice(choice)
                for choice in out.choices
            ],
            usage=out.usage,
            created=out.created,
            model=out.model,
        ).model_dump_json()
    )
