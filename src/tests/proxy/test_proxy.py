from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock


def test_get_models(
    app_server: "TestClient",
    httpx_mock: "HTTPXMock",
    mock_text_model_reference,
):
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/workers",
        match_params={"type": "text"},
        json=[
            {
                "type": "text",
                "name": "worker-1",
                "id": "worker-1-id",
                "online": True,
                "requests_fulfilled": 100,
                "kudos_rewards": 5000,
                "performance": "1000",
                "threads": 4,
                "uptime": 99.9,
                "maintenance_mode": False,
                "nsfw": False,
                "trusted": True,
                "flagged": False,
                "uncompleted_jobs": 0,
                "models": ["google/gemma-4-E2B-it"],
                "bridge_agent": "AI_Horde_Node",
                "max_length": 4096,
                "max_context_length": 10000,
            }
        ],
    )

    resp = app_server.get("/v1/models")
    resp.raise_for_status()
    assert resp.json() == {
        "data": [
            {
                "id": "google/gemma-4-E2B-it",
                "name": "google/gemma-4-E2B-it",
                "canonical_slug": "google/gemma-4-E2B-it",
                "context_length": 5904,
                "description": "Instruction tuned version of Gemma 4 in E2B size",
                "hugging_face_id": "google/gemma-4-E2B-it",
                "top_provider": {
                    "name": "worker-1",
                    "context_length": 5904,
                    "max_completion_tokens": 4096,
                },
                "per_request_limits": {
                    "completion_tokens": 4096,
                    "prompt_tokens": 5904,
                },
                "supported_parameters": ANY,
                "architecture": ANY,
                "pricing": ANY,
            }
        ],
    }


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
            "reference": ANY,
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
        "usage": ANY,
        "model": "google/gemma-4-E2B-it",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you?",
                    "thinking": "Test\n",
                    "reasoning_details": [
                        {
                            "text": "Test\n",
                            "type": "reasoning.text",
                        },
                    ],
                },
            },
        ],
    }
