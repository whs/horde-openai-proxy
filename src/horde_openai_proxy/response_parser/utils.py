from typing import Optional

from horde_openai_proxy.types import HordeModelInfo

from . import template


def get_model_response_template(
    model: str, reference: Optional[HordeModelInfo] = None
) -> Optional[dict]:
    lower_model = model.lower()
    lower_base_model = ""

    if reference is not None:
        lower_model = reference.name.lower()
        if reference.baseline:
            lower_base_model = reference.baseline.lower()

    if "cohere" in lower_model or "cohere" in lower_base_model:
        return template.cohere_template
    elif "smollm" in lower_model or "smollm" in lower_base_model:
        return template.smollm_template
    elif "qwen" in lower_model or "qwen" in lower_base_model:
        return template.qwen3_template
    elif "gpt-oss" in lower_model or "gpt-oss" in lower_base_model:
        return template.gpt_oss_template
    elif "gemma" in lower_model or "gemma" in lower_base_model:
        return template.gemma4_template
    elif "ernie" in lower_model or "ernie" in lower_base_model:
        return template.ernie_template

    return None
