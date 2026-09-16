"""Ask vLLM Semantic Router to choose a model; let LangChain call it."""

import httpx
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.language_models import BaseChatModel


class SemanticRouterMiddleware(AgentMiddleware):
    def __init__(self, models: dict[str, BaseChatModel], router_url: str):
        self.models = models
        self.endpoint = router_url.rstrip("/") + "/api/v1/classify/intent"

    async def awrap_model_call(self, request: ModelRequest, handler):
        messages = list(request.messages)
        if request.system_message is not None:
            messages.insert(0, request.system_message)

        # The classification API accepts text, not a tool-call conversation.
        # Normalize developer messages to system; ignore tool messages entirely.
        roles = {
            "system": "system",
            "developer": "system",
            "human": "user",
            "user": "user",
            "ai": "assistant",
            "assistant": "assistant",
        }
        routing_messages = []
        for message in messages:
            role = roles.get(getattr(message, "role", message.type))
            if role is not None and message.text:
                routing_messages.append({"role": role, "content": message.text})

        try:
            # No model-gateway credentials are sent to this routing-only API.
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                response = await client.post(
                    self.endpoint, json={"messages": routing_messages}
                )
                response.raise_for_status()
                result = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            raise RuntimeError(
                "Semantic Router could not return a routing decision. "
                "Check that Docker is running, the router models are ready, "
                "and SEMANTIC_ROUTER_API_URL points to its API port (8080)."
            ) from None

        choice = result.get("recommended_model") if isinstance(result, dict) else None
        if not isinstance(choice, str) or not choice:
            raise RuntimeError(
                "Semantic Router returned no recommended_model. "
                "Check that the YAML decisions include modelRefs and a fallback."
            )
        if choice not in self.models:
            raise RuntimeError(
                f"Semantic Router selected {choice!r}, but the agent only has "
                f"these models configured: {', '.join(sorted(self.models))}. "
                "Make the YAML modelRefs and the agent's model names match."
            )

        # Keep the original system message, tools, and complete tool history.
        return await handler(request.override(model=self.models[choice]))
