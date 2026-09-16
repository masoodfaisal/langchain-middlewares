"""Two memory tools, scoped to the customer supplied in runtime context."""

from dataclasses import dataclass
from uuid import uuid4

from langchain.tools import ToolRuntime, tool


@dataclass
class UserContext:
    customer_id: int | None = None


def preference_namespace(customer_id: int) -> tuple[str, str]:
    return (str(customer_id), "preferences")


@tool
async def remember(fact: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a durable customer preference for future conversations."""
    await runtime.store.aput(
        preference_namespace(runtime.context.customer_id),
        str(uuid4()),
        {"text": fact},
    )
    return f"Saved preference: {fact}"


@tool
async def recall(runtime: ToolRuntime[UserContext]) -> list[str]:
    """List up to 100 saved preferences for the current customer."""
    memories = await runtime.store.asearch(
        preference_namespace(runtime.context.customer_id), limit=100,
    )
    return [memory.value["text"] for memory in memories]
