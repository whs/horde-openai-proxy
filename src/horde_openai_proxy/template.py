from cachetools import cached, LRUCache

from transformers import (
    AutoTokenizer,
    SentencePieceBackend,
    TokenizersBackend,
)


@cached(LRUCache(maxsize=30))
def get_tokenizer(model: str) -> "TokenizersBackend | SentencePieceBackend":
    """
    Get the adjusted tokenizer for the model.
    :param model: Model name
    :return: Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model.removeprefix("https://huggingface.co/")
    )
    return tokenizer
