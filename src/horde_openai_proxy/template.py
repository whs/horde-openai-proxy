from typing import Optional

from cachetools import cached, LRUCache

from transformers import (
    AutoTokenizer,
    SentencePieceBackend,
    TokenizersBackend,
)

from horde_openai_proxy.response_parser.utils import get_model_response_template
from horde_openai_proxy.types import HordeModelInfo


@cached(LRUCache(maxsize=30))
def get_tokenizer(model: str, reference: Optional[HordeModelInfo] = None) -> "TokenizersBackend | SentencePieceBackend":
    """
    Get the adjusted tokenizer for the model.
    :param model: Model name
    :return: Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model.removeprefix("https://huggingface.co/")
    )

    if getattr(tokenizer, "response_template", None) is None and getattr(tokenizer, "response_schema", None) is None:
        response_template = get_model_response_template(model, reference)
        if response_template is not None:
            tokenizer.response_template = response_template

    return tokenizer
