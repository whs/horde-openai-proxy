import asyncio
import json
import time
from typing import List, cast, Optional
from uuid import uuid4

from .model import get_model
from .template import (
    get_tokenizer,
)
from .types import (
    HordeRequest,
    ModelGenerationInput,
    TextGeneration,
)
from .openai_types import ChatCompletionRequest, ChatCompletionResponse


def openai_to_horde(*args, **kwargs) -> HordeRequest:
    return asyncio.run(openai_to_horde_async(*args, **kwargs))


async def openai_to_horde_async(
    request: ChatCompletionRequest,
    max_context_length: Optional[int] = None,
    max_models: int = 10,
) -> HordeRequest:
    """
    Convert an OpenAI request to a Horde request.

    :param request: The OpenAI request
    :param max_context_length: The maximum context length (not applicable to OpenAI and thus a constant)
    :param max_models: Maximum number of active models to allow (to avoid loading excessive model data)
    :return: The Horde request
    """
    all_stops = set()
    model_names = []
    for model_name in cast(list[str], request.models):
        if len(model_names) >= max_models:
            break

        model_name = model_name.strip()
        model_info = await get_model(model_name)
        if model_info is None:
            continue
        if model_info.hf_url is None:
            continue

        # FIXME: We could get_tokenizer in parallel to speedup lookups, but model_names should be sequential
        try:
            tokenizer = await asyncio.to_thread(get_tokenizer, model_info.hf_url, model_info.reference)
        except Exception:  # TODO: Pokemon
            raise ValueError(f"Model {model_name} not known")

        if tokenizer.chat_template is None:
            continue

        model_names.append(model_name)
        # Fetch all stop words which may be used
        # One should not mix base_models, but if one does, at least stop works
        all_stops.add(tokenizer.eos_token)

    if len(model_names) == 0:
        raise ValueError("All requested models are unknown, offline or unsupported")

    # If the last message is assistant, then this is prefill
    is_prefill = request.messages[-1].role == "assistant"

    primary_model = model_names[0]
    primary_model_info = await get_model(primary_model)
    primary_tokenizer = await asyncio.to_thread(
        get_tokenizer, primary_model_info.hf_url, primary_model_info.reference
    )
    prompt = cast(
        str,
        primary_tokenizer.apply_chat_template(
            [v.model_dump() for v in request.messages],
            tools=[v.model_dump() for v in request.tools]
            if request.tools is not None
            else None,
            add_generation_prompt=True,
            continue_final_message=is_prefill,
            tokenize=False,
        ),
    )

    if max_context_length is None:
        max_context_length = len(primary_tokenizer.encode(prompt)) + request.max_tokens

    return HordeRequest(
        prompt=prompt,
        models=model_names,
        timeout=request.timeout,
        params=ModelGenerationInput(
            max_context_length=max_context_length,
            max_length=request.max_tokens,
            n=request.n,
            rep_pen=request.frequency_penalty,
            stop_sequence=request.stop + list(all_stops),
            temperature=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            top_a=request.top_a,
            top_k=request.top_k,
        ),
    )


def completions_to_openai_response(
    completions: List[TextGeneration],
) -> ChatCompletionResponse:
    return asyncio.run(completions_to_openai_response_async(completions))


async def completions_to_openai_response_async(
    completions: List[TextGeneration], prompt: Optional[str] = None
) -> ChatCompletionResponse:
    """
    Convert a list of completions to an OpenAI response.
    :param completions: List of completions
    :return: OpenAI response
    """
    model = await get_model(completions[0].model)

    parsed_responses = None
    if model is not None and model.hf_url is not None:
        tokenizer = await asyncio.to_thread(get_tokenizer, model.hf_url, model.reference)
        prompt_prefix = prompt
        if getattr(tokenizer, "response_template", None) is None:
            prompt_prefix = None

        try:
            parsed_responses = [
                {
                    "finish_reason": "stop",
                    "index": index,
                    "message": _fix_response(completion),
                }
                for index, completion in enumerate(
                    tokenizer.parse_response(
                        [c.text for c in completions], prefix=prompt_prefix
                    )
                )
            ]

            for response in parsed_responses:
                if "tool_calls" in response['message']:
                    response["finish_reason"] = "tool_calls"
        except AttributeError:
            # Not supported
            pass

    if parsed_responses is None:
        parsed_responses = [
            {
                "finish_reason": "stop",
                "index": index,
                "message": {"role": "assistant", "content": completion.text},
            }
            for index, completion in enumerate(completions)
        ]

    return ChatCompletionResponse(
        id=completions[0].uuid,
        choices=parsed_responses,
        created=int(time.time()),
        model=completions[0].model,
        usage={
            "kudos": completions[0].kudos,
        },
    )


def _fix_response(resp: dict) -> dict:
    if "content" not in resp:
        resp["content"] = None

    for tool in resp.get("tool_calls", []):
        if "id" not in tool:
            # Tool call should have ID
            tool["id"] = str(uuid4())
        if tool.get("type", None) == "function" and "function" in tool:
            # Function arguments must be JSON string and not decoded JSON
            if "arguments" in tool["function"] and not isinstance(
                tool["function"]["arguments"], str
            ):
                tool["function"]["arguments"] = json.dumps(
                    tool["function"]["arguments"]
                )

    return resp
