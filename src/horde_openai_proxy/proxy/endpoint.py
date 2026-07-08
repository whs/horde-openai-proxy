import asyncio
import logging
import time
from typing import List

from fastapi import FastAPI, HTTPException, Request

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
    get_model,
    get_model_best_worker,
)
from horde_openai_proxy.openai_types import (
    ModelArchitecture,
    ModelLimit,
    ModelPricing,
    ModelResponse,
    ModelTopProvider,
    OpenAIModel,
)
from horde_openai_proxy.template import get_tokenizer

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


@app.post("/v1/chat/completions")
async def post_chat_completion(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse:
    token = (
        request.headers.get("authorization", "0000000000")
        .lstrip("Bearer ")
        .lstrip("sk-")
    )

    completions = None
    completion_so_far = ""
    completed_tokens = 0
    can_use_splitting = body.n == 1 and not body.allow_downgrade and "," not in body.model
    try:
        horde_request = await openai_to_horde_async(body)
        timeout = time.time() + horde_request.timeout

        for _ in range(10):
            if can_use_splitting:
                best_worker = await get_model_best_worker(horde_request.models[0])
                if best_worker:
                    if completions is not None:
                        horde_request.prompt += completions[0].text
                    horde_request.params.max_length = min(body.max_tokens - completed_tokens, best_worker.max_completion_tokens)
                    # XXX: We currently do not splice prompts to bypass max_context_length_limit
                    horde_request.timeout = timeout - time.time()
                    if horde_request.timeout <= 0:
                        break
                else:
                    can_use_splitting = False
            
            completions = await get_horde_completion_async(
                token,
                horde_request,
                trusted_workers=body.trusted_workers,
                validated_backends=body.validated_backends,
                slow_workers=body.slow_workers,
                allow_downgrade=body.allow_downgrade,
            )

            if not can_use_splitting:
                break

            completion_so_far += completions[0].text

            model_info = await get_model(completions[0].model)
            # All models are guaranteed by openai_to_horde_async to have hf_url
            tokenizer = await asyncio.to_thread(get_tokenizer, model_info.hf_url)

            completion_tokens = len(tokenizer.encode(completions[0].text))
            completed_tokens += completion_tokens

            # We get more or less than requested tokens, so it's not capping
            if completion_tokens != horde_request.params.max_length:
                break
            # We got the tokens requested
            if completed_tokens >= body.max_tokens:
                break
    except ValueError as e:
        logging.warning("Request error", exc_info=e)
        if completions is None:
            raise HTTPException(status_code=406, detail=str(e))
    except TimeoutError:
        if completions is None:
            raise HTTPException(status_code=504, detail="Request timed out")

    if len(completion_so_far) > 0 and len(completions) == 1:
        completions[0].text = completion_so_far

    return await completions_to_openai_response_async(completions, horde_request.prompt)
