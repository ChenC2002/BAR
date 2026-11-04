import json

import pytest

from bar.policy import LLMPolicy, PolicyError
from bar.schemas import Verification


def sdk_policy(message):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    def respond(request):
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            },
        )

    client = openai.OpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    return LLMPolicy("test", client=client)


def test_official_sdk_parses_typed_output_without_network():
    policy = sdk_policy({"role": "assistant", "content": '{"passed":true,"reason":"supported"}'})
    result = policy.complete("verify", {"selected": []}, Verification)
    assert result.passed and result.reason == "supported"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "refusal": "refused"},
        {"role": "assistant", "content": '{"passed":"not-a-bool","reason":"bad"}'},
    ],
)
def test_refusal_and_malformed_output_raise_policy_error(message):
    with pytest.raises(PolicyError):
        sdk_policy(message).complete("verify", {}, Verification)
