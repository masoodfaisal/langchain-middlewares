# Semantic Router music-store agent

One LangChain agent, three tools, and two models selected by a Docker-hosted
[`vllm-sr`](https://pypi.org/project/vllm-sr/0.3.0/) service:

| Request | Model | Work |
| --- | --- | --- |
| `Show my current invoices.` | `gpt-5.4-mini` | Retrieve invoice dates and totals. |
| `Compare available invoices for last three months and recommend the music based on those purchases.` | `gpt-5.6-luna` | Read invoice details, compare purchases, and find music in the catalog. |

These are the keyword baseline's intended routes; the other examples use their
own signals. The gateway must support Chat Completions tool calling for
`gpt-5.4-mini` and Responses API tool calling for `gpt-5.6-luna`.

```text
LangChain agent → middleware → Semantic Router classification API
                            ← recommended_model
                            → selected model at your gateway → tools / answer
```

[`middleware.py`](middleware.py) calls `POST /api/v1/classify/intent` before
each model invocation. It maps `recommended_model` to one of two pre-created
LangChain model clients and overrides the request's model. **LangChain calls
the generation model directly; Semantic Router does not generate the answer.**
The original conversation and tools still go to the selected model unchanged.

This routes a reasoning workload to a more capable model; it does not inject
provider-specific thinking controls. Reasoning effort follows your gateway's
defaults. Luna uses `/v1/responses` because its reasoning mode with function
tools is not supported on `/v1/chat/completions`. LangGraph carries the full
conversation and tool history; `store=False` and encrypted reasoning content
allow Luna to continue across tool calls without relying on stored responses.
Your gateway must expose the exact model names above.

See the [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses)
for stateless reasoning history.

## Choose a routing example

Each YAML is standalone; the agent and tools stay the same. Start with embeddings.

| Configuration | Signal | Local routing model | Example behavior |
| --- | --- | --- | --- |
| [`router.yaml`](router.yaml) | Keywords | None | Words such as `compare` or `recommend` select the capable model. |
| [`router-embeddings.yaml`](router-embeddings.yaml) | Semantic similarity | mmBERT embedding | Match invoice-lookup or purchase-recommendation examples by meaning. |
| [`router-complexity.yaml`](router-complexity.yaml) | Easy/hard similarity margin | Same mmBERT embedding | Easy tasks select mini; medium and hard tasks select luna. |
| [`router-domains.yaml`](router-domains.yaml) | Trained topic classifier | mmBERT domain classifier | Billing topics select mini; the history/other grouping selects luna. |
| [`router-context.yaml`](router-context.yaml) | Estimated message length | None | 2,048–1,000,000 estimated tokens select luna; other lengths select mini. |

## Setup

Use Python 3.14 and uv. From the repository root:

```bash
uv sync
cd semantic-router-agent
cp -n .env.example .env
```

Configure these values in `.env`:

| Variable | Purpose |
| --- | --- |
| `SEMANTIC_ROUTER_API_URL` | Router classification API, normally `http://127.0.0.1:8080` with no `/v1` suffix. |
| `MODEL_GATEWAY_BASE_URL` | Your gateway's OpenAI-compatible URL, reachable from the Python agent; `http://127.0.0.1:4000/v1` is only an example. |
| `OPENAI_API_KEY` | Your generation gateway's key; sent only to that gateway. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Enable and configure normal LangSmith traces. |

The copy command preserves an existing `.env`. If you used the previous demo,
add `SEMANTIC_ROUTER_API_URL` and `MODEL_GATEWAY_BASE_URL` to that file yourself;
the old `SEMANTIC_ROUTER_BASE_URL` is no longer used. The gateway is not bundled.
Router YAML provider URLs are schema-required, unused placeholders in this
classification-only workflow: do not edit them or put gateway keys in YAML.

`CHINOOK_DB_PATH` points to the sibling `langchain-basics` database; another Chinook
SQLite file works too. The connection is read-only; no other agent code is imported.

## Start the router with local routing models

Have Docker running. The `vllm-sr==0.3.0` Python CLI manages containers with a
matching, pinned runtime image. Routing models run locally on CPU inside the
container: no separate embedding server or key is needed. They classify requests;
both generation models remain on your gateway.

```bash
ROUTER_CONFIG=router-embeddings.yaml
uv run vllm-sr validate --config "$ROUTER_CONFIG"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  uv run vllm-sr serve --config "$ROUTER_CONFIG" --minimal \
  --image ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0
  
```

The `env -u` options omit any exported generation keys from this command because
the CLI otherwise passes them into Docker. They do not change your shell or
`.env`; LangGraph loads that file separately for the Python agent.

First startup downloads the image and selected routing model. Allow time and
disk/RAM for the 307M-parameter encoder. The built-in registry resolves these paths:

| Examples | Hugging Face repository | Folder under this demo |
| --- | --- | --- |
| Embeddings, complexity | `llm-semantic-router/mmbert-embed-32k-2d-matryoshka` | `models/mmbert-embed-32k-2d-matryoshka/` |
| Domains | `llm-semantic-router/mmbert32k-intent-classifier-merged` | `models/mmbert32k-intent-classifier-merged/` |

The CLI mounts the git-ignored `models/` folder at `/app/models` in the container.
Keep YAMLs in this directory to share downloads. Keywords and context need no
model. Automatic model downloads are not pinned to a Hugging Face commit.

Minimal mode starts the router and unused Envoy, without a dashboard/observability
stack. The agent calls **port 8080**, not Envoy's generation proxy on port 8899.
YAML also disables response storage, replay, and shared semantic caching.

Keep this demo private: the classification API has no authentication configured.
The agent's loopback URL does not restrict Docker's published ports; do not expose
them to an untrusted network.

Check readiness or model downloads with:

```bash
uv run vllm-sr status
uv run vllm-sr logs router
```

To switch examples, stop the router and restart with another config. Run one at a time:

```bash
uv run vllm-sr stop
ROUTER_CONFIG=router-complexity.yaml
uv run vllm-sr validate --config "$ROUTER_CONFIG"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  uv run vllm-sr serve --config "$ROUTER_CONFIG" --minimal \
  --image ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0
```

## Start the agent

In another terminal, from this folder:

```bash
uv run langgraph dev --port 2025
```

The middleware and database tools are async: use Studio or `graph.ainvoke(...)`
when invoking the agent from Python.

In Studio select `semantic-router-agent` and supply this run context:

```json
{"customer_id": 44, "as_of": "2025-12-22"}
```

Try the two prompts above in separate threads. The reference date is explicit
because the sample database ends in December 2025. The three-month window is
22 September–22 December 2025, inclusive. Customer 44 has invoices **400** and
**411** in that window: 16 tracks totalling **$15.84**, with no October invoice.
"Current invoices" means recorded purchases through that date; Chinook has no
paid/unpaid invoice status. This is a demo identity, not an authentication system.

## Try each signal

**Embeddings:** try `Find my latest music-store receipt` and
`What should I listen to next based on my recent orders?` after the original prompts.
The latter contains none of the baseline's keywords. Semantic similarity to the
examples determines the route. Tune the starting `0.7` thresholds using your runs;
they are not accuracy guarantees. Soft matching is disabled, permitting mini as
the fallback for low-similarity requests.

**Complexity:** use the original invoice and comparison prompts. The threshold
`0.1` is the hard-minus-easy similarity margin, not a probability: above `0.1`
is hard, below `-0.1` is easy, and the middle is medium. Decisions reference
`music-purchase-task:easy`, `:medium`, or `:hard`. Medium/hard select luna.
This estimates task difficulty; it does not enable an LLM's reasoning mode.

**Domains:** try `Show my current invoices` and
`Explain how rock music developed from blues and recommend music from that era`.
This uses broad trained labels: `billing` groups `business`/`economics`, while
`music_discussion` groups `history`/`other`. Descriptions do not train new classes;
`other` also catches non-music topics. Both original prompts can classify as
business, so domain routing does not reliably distinguish this demo's two tasks.

**Context:** paste a long purchase/music brief into a user message and ask for
recommendations. The 2,048 threshold is for demonstration, not model capacity.
The middleware sends system/user/assistant text (developer messages are mapped
to system for classification only); v0.3.0 counts system/assistant
text plus the latest user message, excluding earlier user messages. Tool results,
arguments, and definitions are not sent for classification, so fetching many
invoices does not itself trigger escalation. Counts estimate UTF-8 bytes / 4,
rounded up. Direct generation does not return usage to the router for calibration.

## Tools and traces

The selected tools are adapted from `langchain-basics`:

- `list_my_invoices`: invoice headers, optionally restricted to the last N months.
- `get_invoice_details`: purchased tracks, artists, albums, and genres.
- `popular_in_genre`: catalog candidates excluding already-purchased tracks and videos.

The capable model compares the tool results and explains its recommendations.
There is no separate music-expert model, Guardian, memory layer, or test suite.

LangChain sends normal model and tool traces to the configured LangSmith project.
Each generation span uses the selected model's name, not `auto`. The routing
HTTP request is not automatically a LangSmith model span; no custom routing,
reasoning, token-usage, or latency display is added.

The classification API uses the first model reference of the matched decision;
each example has exactly one per decision. Proxy plugins and advanced selection
among several model references do not run in this workflow.

Keyword, embedding, complexity, and domain signals use the latest user message;
tool follow-ups normally retain its route. Context uses the text described above.
API failures and missing/unknown model names raise a clear error. An embedding
log with `matched=true` confirms a signal match; the API's `recommended_model`
must also exactly match a name registered in `semantic-router-agent.py`. If the
agent rejects a selection, the error lists the returned and configured names.
Restart the agent after changing its model names, and restart the router after
changing the active YAML. The YAML fallback still selects mini when no
configured signal matches.

## Vela compatibility

These examples use supported mmBERT models, not Vela. Source inspection found
that Vela requests YaRN position scaling and its domain classifier requests CLS
pooling, while the pinned v0.3.0 Candle loaders use different behavior. This is
a compatibility concern, **not an observed Vela startup failure**. Docker or
moving model selection into LangChain does not change those loaders. If Vela
is essential, verify a specific compatible image and its configuration first.

Sources: [Vela domain config](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Domain/raw/main/config.json),
[v0.3.0 domain loader](https://github.com/vllm-project/semantic-router/blob/v0.3.0/candle-binding/src/model_architectures/traditional/modernbert.rs#L818),
[embedding loader](https://github.com/vllm-project/semantic-router/blob/v0.3.0/candle-binding/src/model_architectures/embedding/mmbert_embedding.rs#L198).

All five YAMLs pass the installed `vllm-sr` validator and their model names match
the agent registry. A live embedding classification of `What should I listen to
next based on my recent orders?` selected `purchase-recommendations` and
`gpt-5.6-luna`. One-off checks with mocked HTTP responses verified both model
choices, mini's Chat Completions path, Luna's Responses tool-call round trip with
encrypted reasoning history, and missing/unknown model errors. Earlier checks
covered native LangChain model callbacks and real read-only invoice/catalog
tools. No test files were added. Live generation through the gateway and delivery
to LangSmith have not been verified here.
