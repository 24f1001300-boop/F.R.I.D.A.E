"""Evaluation harness for Project 1.

Calls the agent directly through the gated /debug/ask endpoint (bypassing
Telegram, which cannot be automated) and checks each answer against the
case's expected value.

Comparison is a normalized string match (case/whitespace-insensitive), not a
type-aware comparison. Test cases return heterogeneous answer shapes (plain
numbers, lists, "A=15,B=10"-style strings, ratios) and the bot is free to
format numbers as either numbers or strings, so exact structural comparison
would produce false negatives. This is a known, deliberate limitation: a
"pass" here means the rendered answer matches the expected value textually,
not that every type/shape nuance was validated.

Requires:
  EVAL_BASE_URL  - the running service's base URL (local or deployed)
  EVAL_TOKEN     - must match the server's EVAL_TOKEN env var; the server
                   refuses all /debug/ask requests if EVAL_TOKEN isn't set.
"""
import json
import os
import time
from pathlib import Path
import requests

CASES = json.loads(Path(__file__).with_name("test_cases.json").read_text())
BASE_URL = os.environ.get("EVAL_BASE_URL", "").rstrip("/")
EVAL_TOKEN = os.environ.get("EVAL_TOKEN", "")

if not BASE_URL:
    raise SystemExit("Set EVAL_BASE_URL to the running service's URL before running evaluation.")
if not EVAL_TOKEN:
    raise SystemExit("Set EVAL_TOKEN to match the service's EVAL_TOKEN env var before running evaluation.")


def normalize(value) -> str:
    return str(value).strip().lower().replace(" ", "")


def main():
    print(f"Running {len(CASES)} evaluation cases against {BASE_URL}")

    r = requests.get(f"{BASE_URL}/health", timeout=30)
    r.raise_for_status()
    print("Health:", r.json())

    results = []
    passed = 0
    for i, case in enumerate(CASES):
        started = time.perf_counter()
        status, got = "error", None
        try:
            resp = requests.post(
                f"{BASE_URL}/debug/ask",
                headers={"x-eval-token": EVAL_TOKEN},
                json={"question": case["question"], "chat_id": -100000 - i},
                timeout=240,
            )
            resp.raise_for_status()
            parsed = json.loads(resp.json()["reply"])
            got = parsed.get("answer")
            status = "pass" if normalize(got) == normalize(case["expected"]) else "fail"
        except Exception as e:
            got = f"<error: {e}>"
            status = "error"

        latency = round(time.perf_counter() - started, 4)
        if status == "pass":
            passed += 1
        results.append({
            "id": case["id"], "expected": case["expected"], "got": got,
            "status": status, "latency_s": latency,
        })
        print(f"[{status.upper():5}] {case['id']:16} expected={case['expected']!r} got={got!r} ({latency}s)")

    print(f"\nPassed {passed}/{len(CASES)}")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
