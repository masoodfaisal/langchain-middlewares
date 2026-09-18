"""Granite assesses a memory; Rego decides; middleware gates the tool call."""

import asyncio
import json
import logging
import os
from pathlib import Path
import re
from typing import Annotated, NotRequired

import httpx
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.messages import HumanMessage, ToolMessage
from langsmith import get_current_run_tree
from regopy import Interpreter, LogLevel

from tools import preference_namespace


class MemoryState(AgentState):
    source_user_message: NotRequired[Annotated[str, PrivateStateAttr]]


class MemoryGuardian(AgentMiddleware):
    state_schema = MemoryState

    def __init__(self):
        self.policy = Path(__file__).with_name("guardian.rego").read_text()

    def before_agent(self, state, runtime):
        # Capture this turn's input before the model or tools can add messages.
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        text = last.content if isinstance(last, HumanMessage) else ""
        return {"source_user_message": text if isinstance(text, str) else ""}

    def evaluate(self, query, packet):
        # Each evaluation gets its own interpreter and the same policy snapshot.
        engine = Interpreter()
        engine.log_level = LogLevel.NONE
        engine.strict_built_in_errors = True
        engine.add_module("guardian", self.policy)
        engine.set_input_term(json.dumps(packet, allow_nan=False))
        result = engine.query(f"decision = ({query})")
        if not result.ok() or len(result) != 1:
            raise ValueError("Invalid policy result")
        return result[0].bindings["decision"]

    async def assess_memory(self, packet, criterion):
        messages = [
            {"role": "user", "content": json.dumps({
                "source_user_message": packet["source_user_message"],
                "tool_description": "remember(fact) saves a durable customer preference.",
            })},
            {"role": "assistant", "content": json.dumps(packet["tool"])},
            {"role": "user", "content": (
                "<no-think>As a judge agent, assess whether the provided text "
                "meets the given judging criteria using all available information, "
                "including conversations, documents, and tools. Provide your score "
                "immediately without explanation. Output empty <think>\\n</think> "
                "tags followed by your score in <score></score> tags."
                f"\n\n### Criteria: {criterion}\n\n### Scoring Schema: "
                "If the last assistant's text meets the criteria, return 'yes'; "
                "otherwise, return 'no'."
            )},
        ]
        headers = {}
        if os.getenv("GUARDIAN_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['GUARDIAN_API_KEY']}"
        async with httpx.AsyncClient(
            base_url=os.getenv("GUARDIAN_BASE_URL", "http://127.0.0.1:11434/v1/"),
            headers=headers,
            timeout=60,
        ) as client:
            response = await client.post("chat/completions", json={
                "model": os.getenv("GUARDIAN_MODEL", "guardian-local"),
                "messages": messages,
                "temperature": 0,
                "max_tokens": 128,
            })
        response.raise_for_status()
        choice = response.json()["choices"][0]
        score = re.fullmatch(
            r"\s*(?:<think>\s*</think>\s*)?<score>\s*(yes|no)\s*</score>\s*",
            choice["message"]["content"],
        )
        if choice.get("finish_reason") != "stop" or score is None:
            raise ValueError("Invalid Granite score")
        return score[1] == "yes"

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] not in {"remember", "recall"}:
            return await handler(request)

        reason = "the request does not meet the customer-preference policy."
        try:
            customer_id = getattr(request.runtime.context, "customer_id", None)
            if type(customer_id) is not int or customer_id <= 0:
                raise ValueError("A positive customer_id is required in context")
            if request.runtime.store is None:
                raise ValueError("A runtime memory store is required")
            packet = {
                "customer_id": customer_id,
                "namespace": list(preference_namespace(customer_id)),
                "source_user_message": request.state.get("source_user_message", ""),
                "tool": {"name": call["name"], "args": call["args"]},
                "intent_match": False,
            }
            async with asyncio.timeout(65):
                valid = await asyncio.to_thread(
                    self.evaluate, "data.guardian.valid_request", packet,
                )
                if valid is True and call["name"] == "remember":
                    enabled = os.getenv("ENABLE_GUARDIAN", "true").strip().lower()
                    if enabled == "false":
                        packet["intent_match"] = True
                    else:
                        criterion = await asyncio.to_thread(
                            self.evaluate, "data.guardian.criterion", packet,
                        )
                        packet["intent_match"] = await self.assess_memory(packet, criterion)
                allowed = await asyncio.to_thread(
                    self.evaluate, "data.guardian.allow", packet,
                )
        except Exception as exc:
            logging.getLogger(__name__).warning("Memory check failed: %s", type(exc).__name__)
            allowed = False
            reason = "the memory check could not be completed. Check the demo configuration."

        if allowed is True:
            return await handler(request)
        if run := get_current_run_tree():
            run.add_metadata({"guardian_blocked": reason})
        return ToolMessage(
            content=f"Memory action blocked: {reason}",
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )
