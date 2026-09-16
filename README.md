# LangChain middleware demos

This is a collection of demos that helps with production patterns for implementing least agency, automatic model selection for cost and availability.

The [Guardian demo](guardian/README.md) is a small customer-preferences agent showcasing policy capabilities that sits on top of agentic reasoning layer and applying least agency. **Showcases how to intercept agentic actions before execution and apply deterministic controls.**

Granite model (or gpt safeguard model) works as Guardian which assesses proposed memories (or others such as customer intent), an OPA Rego policy decides whether to
allow them, and LangChain middleware checks the decision before the memory tool
runs. All demo code and configuration live in [`guardian/`](guardian/).

The [Semantic Router music-store demo](semantic-router-agent/README.md) showcases how semantic router helps to dynamically select model based on context, and embeddings and how LangChain Middleware automatically switches model based on recommendaitions from the semantic router.
The agent routes
invoice lookups to `gpt-5.4-mini` and purchase-based music recommendations to
`gpt-5.6-luna`. LangChain middleware asks a Docker-hosted `vllm-sr` instance
for the model choice, then LangChain calls that model directly. The router uses
supported mmBERT models for its learned signals. Its agent, middleware, tools,
and configuration live separately in [`semantic-router-agent/`](semantic-router-agent/).

The [Switchyard music-store demo](switchyard-agent/README.md) uses NVidia SwitchYard llm_classifier approach to ask a judge model to recommend agentic model resulting in lower cost and improved outcomes. The agent uses the same invoice
and recommendation use case with NVIDIA's Switchyard embedded in LangChain
middleware. A `gpt-5.5` judge classifies each task, Switchyard chooses the efficient
or capable model, and LangChain calls it directly. No router server is needed.
