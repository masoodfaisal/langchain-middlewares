# Switchyard music-store agent

A separate version of the invoice/music demo: NVIDIA Switchyard runs **embedded
inside a LangChain middleware** and uses `gpt-5.5` to judge which answer model to
select. LangChain calls the selected model directly. There is no Docker service,
routing API, local embedding model, or model download for this demo.

| Request | Intended answer model | Work |
| --- | --- | --- |
| `Show my current invoices.` | `gpt-5.4-mini` | Retrieve invoice dates and totals. |
| `Compare available invoices for last three months and recommend the music based on those purchases.` | `gpt-5.6-luna` | Compare purchases and recommend catalog music. |

These are the capability prompt's intended routes, not guaranteed classifier
outputs or measured limits of either model.

```text
LangChain agent → Switchyard middleware → gpt-5.5 judge
                                       ← capability verdict
                                       → Switchyard selects an answer model
                → selected model at your gateway → tools / answer
```

## Setup

Use Python 3.14 and uv. From the repository root:

```bash
uv sync --extra switchyard
cd switchyard-agent
cp -n .env.example .env
```

The optional `switchyard` extra installs the official, pinned
[`nemo-switchyard==0.2.0`](https://pypi.org/project/nemo-switchyard/0.2.0/)
package. The existing Guardian and Semantic Router demos are unchanged. The copy
command preserves an existing `.env`; fill in its settings yourself:

| Variable | Purpose |
| --- | --- |
| `MODEL_GATEWAY_BASE_URL` | Your OpenAI-compatible gateway URL; `http://127.0.0.1:4000/v1` is only an example. |
| `OPENAI_API_KEY` | Gateway key, used by the judge and both answer models. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Enable and configure normal LangSmith traces. |
| `CHINOOK_DB_PATH` | Path to the Chinook SQLite database from `langchain-basics`. |

The gateway is not bundled. It must expose the exact model names `gpt-5.5`,
`gpt-5.4-mini`, and `gpt-5.6-luna`. The mini answer model needs Chat Completions
tool calling; Luna needs Responses API tool calling to retain reasoning. The
judge needs Chat Completions JSON-schema structured output. These
names may be gateway aliases: this demo does not check public model availability.
The database connection is read-only, and no sibling agent code is imported.

Answer calls use `store=False`, with conversation history managed by LangGraph.
Luna also requests encrypted reasoning content so LangChain can carry it across
tool calls, following [OpenAI's reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning#keeping-reasoning-items-in-context).

Start the agent from this folder:

```bash
uv run --extra switchyard langgraph dev --port 2026
```

Keep `--extra switchyard` on `uv run` commands so uv retains the optional package.
Select `switchyard-agent` in Studio and supply this run context:

```json
{"customer_id": 44, "as_of": "2025-12-22"}
```

Try the two prompts above in separate threads. The middleware and tools are
async; use Studio or `graph.ainvoke(...)` when calling the agent from Python.

The sample data ends in December 2025. The three-month window is 22 September–
22 December 2025, inclusive. Customer 44 has invoices **400** and **411** in this
window: 16 tracks totalling **$15.84**, with no October invoice. "Current invoices"
means recorded purchases through the reference date, not unpaid bills. Customer
identity comes from runtime context; this demo is not an authentication system.

## How the classifier works

[`middleware.py`](middleware.py) implements the capability-classification approach
in NVIDIA's [LLM classifier guide](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/docs/routing_algorithms/llm_classifier_routing.md).
The released 0.2.0 Python factory is named `algorithms.llm_task_classifier`.
The guide on `main` describes newer interfaces; this example uses the
[tagged 0.2.0 bindings](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.2.0/switchyard_rust/libsy.py).

The judge estimates whether the efficient agent can complete the whole task,
using a small capability card in `CLASSIFIER_PROMPT`. Invoice retrieval and
single-invoice arithmetic are supported; cross-invoice comparisons and inferred
music preferences are limitations. Ambiguous tasks fall into uncertain/unmatched
categories. Edit that prompt to change the demo rubric.

Switchyard compares the judge's `p_solve` estimate with a category-adjusted
threshold. With base threshold `0.5` and `threshold_step=0.1`:

| Judge category | Select efficient when |
| --- | --- |
| `supported` | `p_solve >= 0.5` |
| `uncertain` or `unmatched` | `p_solve >= 0.6` |
| `unsupported` | `p_solve >= 0.7` |

Otherwise Switchyard selects the capable model. An invalid judge verdict or a
judge-call failure also selects the capable model. Thus `unsupported` does not
unconditionally select capable: it raises the score required for efficient.
The score is a judge estimate, not a calibrated success probability.

The middleware sends only user text for classification; Switchyard's classifier
builds its task brief from the opening and latest user turns. It does not inspect
the agent's tool results. The original system prompt, messages, tool definitions,
and tool results remain intact on the selected model's LangChain request.

In 0.2.0, Switchyard's managed `run()` also invokes its selected target. The two
small selection callbacks return a local acknowledgement, **not a generated
answer**. The middleware reads the resulting model choice and calls LangChain's
handler with `request.override(model=...)` for the real generation call.

## Tools, traces, and tradeoffs

The copied tools remain simple and read-only:

- `list_my_invoices`: invoice headers, optionally restricted to the last N months.
- `get_invoice_details`: purchased tracks, artists, albums, and genres.
- `popular_in_genre`: catalog music excluding already-purchased tracks and videos.

Each model invocation creates a fresh embedded router with
`session_affinity=False`. The judge runs again on tool continuations, even when
the user text has not changed. This keeps the example small, but adds judge
requests, latency, and cost; routing does not guarantee savings. There is no
routing cache, training step, Guardian memory layer, or test suite.

The judge and answer models use normal `ChatOpenAI` calls, so LangSmith captures
their model traces alongside the tool traces. The judge carries the `nostream`
tag to keep its classification JSON out of LangGraph's chat stream; it remains
traceable, and `disable_streaming=True` makes its upstream JSON response
non-streaming too. No custom routing, reasoning, usage, or latency display is added.
Reasoning settings follow the gateway defaults.

The pinned package is installed and its native runtime was verified with mocked
model HTTP responses. One-off checks covered both model choices, judge-error and
invalid-verdict fallbacks, the real read-only tools, concurrent runs, parented
LangChain model callbacks, and keeping judge JSON out of the chat stream. Python
compilation and installed-dependency compatibility checks also passed. No test
files were added. Mocked HTTP checks also covered Responses tool loops, encrypted
reasoning replay, and switching between the two answer models in both directions.
Live gateway calls, judge accuracy, and delivery to LangSmith
have not been verified.
