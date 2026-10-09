"""Offline regression checks: python test_research.py."""
from types import SimpleNamespace
from unittest.mock import Mock

from research import complete_report, configure_local_model
from unittest.mock import patch
from agents import bounded_messages, call_with_tool_correction
from langchain.agents.middleware import ModelRequest
from openai import BadRequestError
import httpx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def run():
    for endpoint, expected in [("http://localhost:11434/v1", 600), ("https://api.groq.com/openai/v1", 120)]:
        from langchain_openai import ChatOpenAI
        original = ChatOpenAI(model="local-test", api_key="not-a-secret", base_url=endpoint, timeout=120)
        with patch.dict("os.environ", {"LAB_BASE_URL": endpoint}):
            model = configure_local_model(original)
        assert model.request_timeout == expected
        assert model.root_client.timeout == expected
        assert model.root_async_client.timeout == expected
        assert model.client is model.root_client.chat.completions
        assert model.async_client is model.root_async_client.chat.completions
        assert original.root_client.timeout == 120
        if expected == 600:
            assert model.root_client.max_retries == 0
            assert model.root_async_client.max_retries == 0
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.com"))
    rejected = BadRequestError("invalid tool", response=response, body={
        "code": "tool_use_failed", "message": "attempted to call tool 'exec' which was not in request.tools"})
    request = ModelRequest(model=Mock(), messages=[], tools=[{"type": "function", "function": {"name": "execute"}}])
    handler = Mock(side_effect=[rejected, "ok"])
    assert call_with_tool_correction(request, handler) == "ok"
    assert handler.call_count == 2
    assert "execute" in handler.call_args.args[0].system_message.content
    handler = Mock(side_effect=rejected)
    try:
        call_with_tool_correction(request, handler)
    except BadRequestError:
        assert handler.call_count == 2
    else:
        raise AssertionError("correction must be bounded to one retry")
    unrelated = BadRequestError("bad input", response=response, body={"code": "invalid_request_error"})
    handler = Mock(side_effect=unrelated)
    try:
        call_with_tool_correction(request, handler)
    except BadRequestError:
        assert handler.call_count == 1
    else:
        raise AssertionError("unrelated errors must not be retried")
    history = [HumanMessage(content="topic"), AIMessage(content="old" * 1000),
               AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "a"}]),
               ToolMessage(content="saved notes", tool_call_id="a")]
    trimmed = bounded_messages(history, 2000)
    assert trimmed == [history[0], *history[2:]]
    assert sum(len(m.model_dump_json().encode("utf-8")) for m in trimmed) <= 2000
    assert bounded_messages(history, 20000) == history
    try:
        bounded_messages(history, 10)
    except RuntimeError:
        pass
    else:
        raise AssertionError("oversized initial task must fail")
    large = [history[0], history[2], ToolMessage(content="evidence 🐎" * 10000, tool_call_id="a")]
    preview = bounded_messages(large, 2000)
    assert sum(len(m.model_dump_json().encode("utf-8")) for m in preview) <= 2000
    assert preview[1].tool_calls == large[1].tool_calls
    assert preview[2].tool_call_id == "a"
    assert "PREVIEW ONLY" in preview[2].content
    assert large[2].content == "evidence 🐎" * 10000
    parallel = [history[0], AIMessage(content="", tool_calls=[
        {"name": "task", "args": {}, "id": str(i)} for i in range(3)
    ]), *[ToolMessage(content="result" * 10000, tool_call_id=str(i)) for i in range(3)]]
    preview = bounded_messages(parallel, 3000)
    assert len(preview) == 5
    assert sum(len(m.model_dump_json().encode("utf-8")) for m in preview) <= 3000
    assert [m.tool_call_id for m in preview[2:]] == ["0", "1", "2"]
    message = SimpleNamespace(content="Finished", tool_calls=[{"name": "task"}])
    result = {"messages": [message]}
    ok = SimpleNamespace(exit_code=0, output="OK")
    missing = SimpleNamespace(exit_code=1, output="missing")

    backend, agent = Mock(), Mock()
    backend.execute.return_value = ok
    assert complete_report(agent, backend, result) == [message]
    agent.invoke.assert_not_called()

    backend.execute.side_effect = [missing, ok]
    agent.invoke.return_value = result
    assert complete_report(agent, backend, result) == [message]
    assert agent.invoke.call_count == 1

    backend.execute.side_effect = [missing]
    no_tools = {"messages": [SimpleNamespace(content="I cannot use tools", tool_calls=[])]}
    try:
        complete_report(agent, backend, no_tools)
    except RuntimeError as exc:
        assert "no tool calls" in str(exc)
    else:
        raise AssertionError("text-only model must fail explicitly")

    backend.execute.side_effect = [missing, missing]
    try:
        complete_report(agent, backend, result)
    except RuntimeError as exc:
        assert "validation failed" in str(exc)
    else:
        raise AssertionError("unfinished continuation must fail")
    assert agent.invoke.call_count == 2
    backend.execute.side_effect = [missing, ok]
    malformed = {"messages": [SimpleNamespace(content="", tool_calls=[], response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"})]}
    assert complete_report(agent, backend, malformed) == [message]
    assert agent.invoke.call_count == 3
    print("OK: completion, bounded continuation, text-only model, malformed call, validation failure")


if __name__ == "__main__":
    run()
