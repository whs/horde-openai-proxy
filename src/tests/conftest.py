import pytest
from pytest_httpx import HTTPXMock


@pytest.fixture
def mock_horde_model(httpx_mock: "HTTPXMock"):
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/status/models",
        match_params={"type": "text", "min_count": "1"},
        json=[
            {
                "performance": 10,
                "queued": 0,
                "jobs": 0,
                "eta": 0,
                "type": "text",
                "name": "google/gemma-4-E2B-it",
                "count": 1,
            }
        ],
        is_optional=True,
    )


@pytest.fixture
def mock_text_model_reference(httpx_mock: "HTTPXMock"):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/db0/AI-Horde-text-model-reference/main/db.json",
        json={
            "google/gemma-4-E2B-it": {
                "name": "google/gemma-4-E2B-it",
                "model_name": "gemma-4-E2B-it",
                "baseline": "",
                "parameters": 2000000000,
                "description": "Instruction tuned version of Gemma 4 in E2B size",
                "version": "1",
                "style": "generalist",
                "nsfw": False,
                "display_name": "Gemma4 E2B IT",
                "url": "https://huggingface.co/google/gemma-4-E2B-it",
                "tags": ["4B"],
            },
        },
        is_optional=True,
    )
