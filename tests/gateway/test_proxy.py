from __future__ import annotations

import asyncio
import json

import httpx

from polar.gateway.engine import SGLangEngine
from polar.gateway.proxy import InferenceClient


def test_sglang_completion_uses_configured_router_and_preserves_token_extensions() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.host or "", request.url.path))
        if request.url.host == "router" and request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["return_prompt_token_ids"] is True
            assert body["return_meta_info"] is True
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "message": {"role": "assistant", "content": "x"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [{"token": "x", "logprob": -0.1, "bytes": [120]}]
                            },
                            "meta_info": {
                                "output_token_logprobs": [[-0.1, 4, "x"]]
                            },
                        }
                    ]
                },
            )
        return httpx.Response(500, json={"error": f"unexpected {request.url}"})

    async def run() -> dict:
        client = InferenceClient("http://router:9000", SGLangEngine())
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.completion({"messages": [{"role": "user", "content": "hi"}]})
        finally:
            await client.close()

    response = asyncio.run(run())
    choice = response["choices"][0]

    # Worker selection belongs to the configured router. The Slime router patch
    # forwards the token extensions; bypassing it here loses its scheduling semantics.
    assert requests == [("router", "/v1/chat/completions")]
    assert choice["input_token_ids"] == [1, 2, 3]
    assert choice["token_ids"] == [4]
    assert choice["logprobs"]["content"][0]["token_id"] == 4
