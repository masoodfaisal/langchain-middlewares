"""OpenJev contract and memory-policy regressions, without a running model."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from langchain.messages import HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore


GUARDIAN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARDIAN_DIR))
spec = importlib.util.spec_from_file_location(
    "guardian_openjev_test_module", GUARDIAN_DIR / "guardian-openjev.py",
)
openjev = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {
    "OPENAI_API_KEY": "test-only",
    "LANGSMITH_TRACING": "false",
}):
    spec.loader.exec_module(openjev)

from tools import UserContext, preference_namespace


ASYNC_CLIENT = httpx.AsyncClient


def decision(choice="allow", allow=0.9, deny=0.1):
    return {"answers": {"memory": {
        "type": "choice",
        "choice": choice,
        "probabilities": {"allow": allow, "deny": deny},
    }}}


class OpenJevTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only",
            "LANGSMITH_TRACING": "false",
            "ENABLE_GUARDIAN": "true",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.guardian = openjev.OpenJevMemoryGuardian()
        self.store = InMemoryStore()

    def request(self, name="remember", args=None, customer_id=42,
                source="I prefer email contact.", store=True):
        if args is None:
            args = {"fact": "Prefers email contact."} if name == "remember" else {}
        return SimpleNamespace(
            tool_call={"id": "test-call", "name": name, "args": args},
            runtime=SimpleNamespace(
                context=UserContext(customer_id=customer_id),
                store=self.store if store else None,
            ),
            state={
                "source_user_message": source,
                "messages": [HumanMessage(content="unrelated private history")],
            },
        )

    def memory_handler(self):
        async def execute(request):
            namespace = preference_namespace(request.runtime.context.customer_id)
            if request.tool_call["name"] == "remember":
                await request.runtime.store.aput(
                    namespace, "saved", {"text": request.tool_call["args"]["fact"]},
                )
                text = "Saved preference."
            else:
                memories = await request.runtime.store.asearch(namespace)
                text = json.dumps([item.value["text"] for item in memories])
            return ToolMessage(content=text, tool_call_id="test-call")

        return AsyncMock(side_effect=execute)

    @contextmanager
    def judge(self, response=None, *, status=200, error=None, raw=None):
        requests = []

        def respond(request):
            requests.append(request)
            if error is not None:
                raise error("Judge unavailable", request=request)
            return httpx.Response(
                status,
                content=raw if raw is not None else json.dumps(response),
                headers={"Content-Type": "application/json"},
            )

        def client(**kwargs):
            return ASYNC_CLIENT(transport=httpx.MockTransport(respond), **kwargs)

        with patch.object(openjev.httpx, "AsyncClient", side_effect=client):
            yield requests

    async def assert_blocked(self, request, **judge_kwargs):
        handler = self.memory_handler()
        with self.judge(**judge_kwargs) as requests:
            result = await self.guardian.awrap_tool_call(request, handler)
        handler.assert_not_awaited()
        self.assertEqual(result.status, "error")
        self.assertEqual(await self.store.asearch(preference_namespace(42)), [])
        return requests

    async def test_allow_saves_for_the_current_customer_only(self):
        handler = self.memory_handler()
        with self.judge(decision()) as requests:
            result = await self.guardian.awrap_tool_call(self.request(), handler)
        handler.assert_awaited_once()
        self.assertEqual(result.content, "Saved preference.")
        self.assertEqual(len(requests), 1)
        saved = await self.store.asearch(preference_namespace(42))
        self.assertEqual([item.value["text"] for item in saved], ["Prefers email contact."])
        self.assertEqual(await self.store.asearch(preference_namespace(43)), [])

    async def test_deny_does_not_save(self):
        requests = await self.assert_blocked(
            self.request(), response=decision("deny", allow=0.1, deny=0.9),
        )
        self.assertEqual(len(requests), 1)

    async def test_native_request_contains_only_the_current_fact_and_source(self):
        criterion = self.guardian.evaluate("data.guardian.criterion", {})
        packet = {
            "customer_id": 42,
            "namespace": ["42", "preferences"],
            "source_user_message": "I prefer email contact.",
            "tool": {"name": "remember", "args": {"fact": "Prefers email contact."}},
            "intent_match": False,
            "messages": [{"content": "unrelated private history"}],
        }
        with self.judge(decision()) as requests:
            self.assertTrue(await self.guardian.assess_memory(packet, criterion))
        request = requests[0]
        self.assertEqual(str(request.url), "http://127.0.0.1:8080/v1/systemone")
        self.assertEqual(request.method, "POST")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "openjev")
        self.assertEqual(payload["state"], {
            "source_user_message": packet["source_user_message"],
            "tool_description": "remember(fact) saves a durable customer preference.",
            "tool": packet["tool"],
        })
        self.assertEqual(set(payload), {"model", "state", "questions"})
        self.assertEqual(set(payload["questions"]), {"memory"})
        question = payload["questions"]["memory"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["instructions"], criterion)
        self.assertEqual(set(question["criteria"]), {"allow", "deny"})
        self.assertNotIn("authorization", request.headers)

    async def test_openjev_settings_are_independent_of_granite(self):
        with patch.dict(os.environ, {
            "OPENJEV_BASE_URL": "http://judge.example/v1/",
            "OPENJEV_MODEL": "openjev-custom",
            "OPENJEV_API_KEY": "openjev-test-only",
            "GUARDIAN_BASE_URL": "http://granite.example/v1/",
            "GUARDIAN_MODEL": "granite-custom",
            "GUARDIAN_API_KEY": "granite-test-only",
        }):
            handler = self.memory_handler()
            with self.judge(decision()) as requests:
                await self.guardian.awrap_tool_call(self.request(), handler)
        handler.assert_awaited_once()
        request = requests[0]
        self.assertEqual(str(request.url), "http://judge.example/v1/systemone")
        self.assertEqual(json.loads(request.content)["model"], "openjev-custom")
        self.assertEqual(request.headers["authorization"], "Bearer openjev-test-only")
        self.assertNotIn("granite", request.content.decode())

    async def test_malformed_decisions_block_saves(self):
        cases = {
            "missing answers": {},
            "wrong answer type": {"answers": {"memory": {**decision()["answers"]["memory"], "type": "rating"}}},
            "missing type": {"answers": {"memory": {"choice": "allow", "probabilities": {"allow": 0.9, "deny": 0.1}}}},
            "missing probabilities": {"answers": {"memory": {"type": "choice", "choice": "allow"}}},
            "unknown choice": decision("yes"),
            "boolean probability": decision(allow=True, deny=0),
            "string probability": decision(allow="0.9"),
            "infinite probability": decision(allow=float("inf")),
            "nan probability": decision(allow=float("nan")),
            "negative probability": decision(allow=-0.1, deny=1.1),
            "above one": decision(allow=1.1, deny=-0.1),
            "not normalized": decision(allow=0.7, deny=0.2),
            "choice contradicts probabilities": decision("allow", allow=0.1, deny=0.9),
            "deny contradicts probabilities": decision("deny", allow=0.9, deny=0.1),
            "allow tie": decision("allow", allow=0.5, deny=0.5),
            "deny tie": decision("deny", allow=0.5, deny=0.5),
        }
        missing_label = decision()
        del missing_label["answers"]["memory"]["probabilities"]["deny"]
        cases["missing probability label"] = missing_label
        extra_label = decision()
        extra_label["answers"]["memory"]["probabilities"]["other"] = 0
        cases["extra probability label"] = extra_label
        for name, response in cases.items():
            with self.subTest(name=name):
                requests = await self.assert_blocked(self.request(), response=response)
                self.assertEqual(len(requests), 1)

    async def test_unavailable_or_invalid_http_response_blocks_saves(self):
        cases = [
            {"error": httpx.ConnectError},
            {"error": httpx.ReadTimeout},
            {"response": decision(), "status": 500},
            {"raw": "not JSON"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                requests = await self.assert_blocked(self.request(), **kwargs)
                self.assertEqual(len(requests), 1)

    async def test_invalid_requests_are_rejected_before_calling_the_judge(self):
        requests = [self.request(customer_id=value) for value in (None, True, 0, -1, "42", 42.0)]
        requests.extend(self.request(args=args) for args in (
            {}, {"fact": " "}, {"fact": 123}, {"fact": "x" * 1001},
            {"fact": "Prefers email contact.", "customer_id": 43},
        ))
        requests.extend((
            self.request(source=""), self.request(source="x" * 8001),
            self.request(source=123), self.request(store=False),
            self.request(name="recall", args={"customer_id": 43}),
        ))
        for request in requests:
            with self.subTest(call=request.tool_call, context=request.runtime.context,
                              source_type=type(request.state["source_user_message"])):
                calls = await self.assert_blocked(request, error=httpx.ConnectError)
                self.assertEqual(calls, [])

    async def test_recall_works_with_offline_judge_and_preserves_customer_scope(self):
        await self.store.aput(preference_namespace(42), "mine", {"text": "Prefers email."})
        await self.store.aput(preference_namespace(43), "other", {"text": "Prefers phone."})
        handler = self.memory_handler()
        with self.judge(error=httpx.ConnectError) as requests:
            result = await self.guardian.awrap_tool_call(self.request(name="recall"), handler)
        handler.assert_awaited_once()
        self.assertEqual(json.loads(result.content), ["Prefers email."])
        self.assertEqual(requests, [])

    async def test_shared_disable_flag_skips_judge_but_keeps_request_validation(self):
        with patch.dict(os.environ, {"ENABLE_GUARDIAN": "false"}):
            handler = self.memory_handler()
            with self.judge(error=httpx.ConnectError) as requests:
                await self.guardian.awrap_tool_call(self.request(), handler)
                blocked = await self.guardian.awrap_tool_call(
                    self.request(args={"fact": ""}), handler,
                )
        handler.assert_awaited_once()
        self.assertEqual(blocked.status, "error")
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
