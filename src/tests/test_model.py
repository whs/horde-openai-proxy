import pytest

from horde_openai_proxy.model import estimate_hf_url, get_references_async


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("google/gemma-4-E2B-it", "https://huggingface.co/google/gemma-4-E2B-it"),
        (
            "koboldcpp/google/gemma-4-E2B-it",
            "https://huggingface.co/google/gemma-4-E2B-it",
        ),
        (
            "aphrodite/google/gemma-4-E2B-it",
            "https://huggingface.co/google/gemma-4-E2B-it",
        ),
        (
            "koboldcpp/digo-prayudha/unsloth-llama-3.2-1b-gguf",
            "https://huggingface.co/digo-prayudha/unsloth-llama-3.2-1b-gguf",
        ),
        (
            "aphrodite/TheDrummer/Cydonia-24B-v4.3",
            "https://huggingface.co/TheDrummer/Cydonia-24B-v4.3",
        ),
        (
            "aphrodite/TheDrummer/Cydonia-24B-v4.3",
            "https://huggingface.co/TheDrummer/Cydonia-24B-v4.3",
        ),
        ("koboldcpp/Qwen_Qwen3-0.6B-IQ4_XS", "https://huggingface.co/Qwen/Qwen3-0.6B"),
        ("koboldcpp/Gemma-4-Harmonia-31B-uncensored-heretic-Q4_K_M", None),
    ],
)
@pytest.mark.asyncio
async def test_estimate_hf_url(
    model_name: str, expected: str, mock_text_model_reference
):
    assert estimate_hf_url(model_name, await get_references_async()) == expected
