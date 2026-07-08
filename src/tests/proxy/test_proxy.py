from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host != "huggingface.co"
)
def test_get_chat_models(
    app_server: "TestClient", mock_horde_model, mock_text_model_reference
):
    resp = app_server.get("/v1/chat/models")
    resp.raise_for_status()
    assert resp.json() == [
        {
            "name": "google/gemma-4-E2B-it",
            "hf_url": ANY,
            "known_to_horde": True,
        },
    ]


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host != "huggingface.co"
)
def test_post_chat_completion(
    app_server: "TestClient",
    httpx_mock: "HTTPXMock",
    mock_horde_model,
    mock_text_model_reference,
):
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/generate/text/async",
        method="POST",
        json={"id": "test-uuid-123"},
    )
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/generate/text/status/test-uuid-123",
        method="GET",
        json={
            "done": False,
            "is_possible": True,
            "faulted": False,
            "kudos": 0,
            "generations": [],
        },
    )
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/generate/text/status/test-uuid-123",
        method="GET",
        json={
            "done": True,
            "is_possible": True,
            "faulted": False,
            "kudos": 42,
            "generations": [
                {
                    "text": "<|channel>thought\nTest\n<channel|>Hello, how can I help you?",
                    "model": "google/gemma-4-E2B-it",
                }
            ],
        },
    )

    resp = app_server.post(
        "/v1/chat/completions",
        json={
            "model": "google/gemma-4-E2B-it",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    resp.raise_for_status()
    data = resp.json()
    assert data == {
        "created": ANY,
        "id": "test-uuid-123",
        "usage": {"kudos": 42},
        "model": "google/gemma-4-E2B-it",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you?",
                    "thinking": "Test\n",
                },
            },
        ],
    }
