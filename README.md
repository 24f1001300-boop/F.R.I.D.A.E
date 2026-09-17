# Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions sent over Telegram for IIT Madras Tools in Data Science Project 1.

## What it does

The bot receives a plain-text analytical question through Telegram, uses an OpenAI-compatible LLM endpoint to reason about the task, and can call a Python analysis tool to fetch public data and compute results with pandas/numpy/requests/BeautifulSoup/openpyxl. It returns exactly one JSON object:

```json
{"answer": {"state": "Assam"}, "log_url": "https://<host>/run.jsonl"}
```

Multi-turn conversations are supported through per-chat history.

## Architecture

- `bot.py`: FastAPI service, Telegram long polling, agent loop, Python analysis tool and public JSONL logging.
- `run.jsonl`: append-only execution trace containing questions, LLM timing, tool calls/results, recovery attempts and final answers.
- `evaluation/`: small reproducible test set and evaluation harness for local/deployed testing.
- Google Cloud Storage can be used as the durable/public location for exported run logs.

### Agent loop

```text
Telegram message
      ↓
FastAPI service / Telegram poller
      ↓
OpenAI-compatible LLM endpoint
      ↓
LLM decides whether Python is needed
      ↓
Python tool → public data acquisition + pandas/numpy computation
      ↓
tool result returned to LLM
      ↓
final structured JSON
      ↓
Telegram
```

### Reliability upgrades

1. **Dataset profiling**: `profile_dataframe(df)` provides deterministic schema, missing-value, duplicate, dtype and numeric-summary information.
2. **AST guardrails**: model-generated Python is parsed and checked before execution. Common filesystem/process/introspection primitives and unapproved imports are blocked.
3. **Bounded recovery**: failed or blocked Python executions trigger at most two corrective attempts.
4. **Latency instrumentation**: LLM calls, Python tool calls and total run latency are recorded in JSONL.
5. **Evaluation layer**: `evaluation/test_cases.json` provides a fixed smoke-test set; the harness checks deployment health and is designed for manual Telegram execution because Telegram is the production interface.

> The AST layer is an application guardrail, **not a security sandbox**. True isolation would require a separate container/VM or sandboxed execution service.

## Run

```bash
pip install -r requirements.txt
export BOT_TOKEN=...          # from @BotFather
export AIPIPE_TOKEN=...       # OpenAI-compatible API token
export BASE_URL=https://your-host
uvicorn bot:app --host 0.0.0.0 --port 8000
```

For the evaluation smoke test:

```bash
export EVAL_BASE_URL=https://your-host
python evaluation/evaluate.py
```
