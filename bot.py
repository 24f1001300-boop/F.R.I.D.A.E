"""Data-analyst Telegram bot — TDS Project 1.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to every message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so the free host never idles out.
"""

import io
import json
import os
import re
import shutil
import threading
import time
import traceback
import contextlib
import ast
import math
import statistics
import socket
import ipaddress
from urllib.parse import urlparse
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
EVAL_TOKEN = os.environ.get("EVAL_TOKEN", "")  # unset => /debug/ask is fully disabled
LOG_PATH = "run.jsonl"
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = 10
PY_TIMEOUT = 60  # seconds for one run_python call
ANSWER_BUDGET = 210  # wall-clock seconds before we force a final answer
MAX_RECOVERY_ATTEMPTS = 2
MAX_CODE_CHARS = 20000
MAX_OUTPUT_CHARS = 8000
FETCH_TIMEOUT = 15          # seconds for one outbound web-fetch
MAX_FETCH_BYTES = 3_000_000 # cap on a single fetched response
MAX_FETCH_REDIRECTS = 3

# Telegram's public Bot API refuses to serve files larger than 20MB via
# getFile (only a self-hosted Local Bot API Server lifts this, which is out
# of scope here). We stay a little under that hard platform ceiling.
MAX_UPLOAD_BYTES = 19_000_000
UPLOAD_ROOT = os.environ.get("UPLOAD_ROOT", "/tmp/fridae_uploads")
UPLOAD_EXPIRY_SECONDS = int(os.environ.get("UPLOAD_EXPIRY_SECONDS", 6 * 3600))
DATASET_CONTEXT_TAG = "[UPLOADED_DATASET_CONTEXT]"

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}  # chat_id -> chat-completion messages
_hist_lock = threading.Lock()

# Uploaded-dataset state: at most one active dataset per chat. Keyed by
# chat_id (always an int from Telegram), never by anything user-supplied.
_uploads: dict[int, dict] = {}
_uploads_lock = threading.Lock()

# Per-chat locks: two messages from the SAME chat arriving close together must
# be processed one after another (otherwise their LLM calls interleave and can
# corrupt turn order in that chat's history). Different chats still run fully
# concurrently — only same-chat requests are serialized.
_chat_locks: dict[int, threading.Lock] = {}
_chat_locks_guard = threading.Lock()


def _get_chat_lock(chat_id: int) -> threading.Lock:
    with _chat_locks_guard:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[chat_id] = lock
        return lock


# ---------------------------------------------------------------- logging
def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- tools
ALLOWED_IMPORTS = {
    "pandas", "numpy", "bs4", "openpyxl", "io", "json",
    "math", "statistics", "re", "datetime", "collections"
}
# "requests" is intentionally NOT allowed here: raw network access from
# model-generated code would bypass the SSRF/size/timeout guardrails below.
# Retrieval must go through the injected fetch_url/fetch_soup/fetch_table/
# fetch_csv/fetch_excel helpers instead (see "web retrieval" section).
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "help", "open", "globals", "locals", "vars", "dir", "getattr",
    "setattr", "delattr",
    # pandas readers that can fetch a URL directly via pandas' own I/O layer,
    # bypassing our guardrails entirely if left open. read_csv/read_json are
    # NOT in this blocklist — they're needed for parsing inline pasted data —
    # but they ARE restricted below (RESTRICTED_TO_BUFFER) to only accept an
    # in-memory io.StringIO/BytesIO buffer, not a path or URL string, which
    # closes the equivalent bypass for those two functions.
    "read_html", "read_excel", "read_sql", "read_sql_query", "read_sql_table",
    "read_parquet", "read_feather", "read_pickle", "read_orc", "read_hdf",
    "read_stata", "read_spss", "read_sas", "read_gbq",
}
BLOCKED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "builtins",
    "importlib", "ctypes", "signal", "pickle", "marshal", "resource"
}
# Dunder attributes are blocked by pattern rather than a fixed list, since a
# hardcoded set (__globals__, __code__, ...) misses anything not enumerated
# (e.g. __reduce__, __getattribute__, __init_subclass__). This is still an
# application-level guardrail, not a sandbox: it cannot catch bypasses built
# from allowed operations at runtime (e.g. constructing an attribute name
# from a string). A small allowlist covers benign, commonly-needed dunders.
SAFE_DUNDERS = {"__name__", "__doc__"}


RESTRICTED_TO_BUFFER = {"read_csv", "read_json"}


def _is_buffer_construction(node) -> bool:
    """True if node is a call to StringIO/BytesIO (bare or io.-qualified)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in {"StringIO", "BytesIO"}
    if isinstance(func, ast.Name):
        return func.id in {"StringIO", "BytesIO"}
    return False


def validate_python(code: str) -> tuple[bool, str]:
    """Validate model-generated Python before execution.

    This is an application-level guardrail, NOT a security sandbox. It blocks
    common filesystem/process/introspection primitives while preserving the
    data-analysis/network workflow required by the project.
    """
    if not isinstance(code, str) or not code.strip():
        return False, "empty Python code"
    if len(code) > MAX_CODE_CHARS:
        return False, f"code exceeds {MAX_CODE_CHARS} characters"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"blocked import: {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"blocked import: {root}"
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                return False, f"blocked name: {node.id}"
        elif isinstance(node, ast.Call):
            call_name = None
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            if call_name in BLOCKED_CALLS:
                kind = "call" if isinstance(node.func, ast.Name) else "attribute call"
                return False, f"blocked {kind}: {call_name}"
            if call_name in RESTRICTED_TO_BUFFER:
                first_arg = node.args[0] if node.args else None
                if first_arg is None:
                    for kw in node.keywords:
                        if kw.arg in {"filepath_or_buffer", "path_or_buf"}:
                            first_arg = kw.value
                if not _is_buffer_construction(first_arg):
                    return False, (
                        f"{call_name} must be called on an io.StringIO/BytesIO buffer, "
                        f"not a path or URL — use fetch_csv/fetch_url for remote data"
                    )
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__") and attr not in SAFE_DUNDERS:
                return False, f"blocked dunder attribute: {attr}"
    return True, "ok"


def profile_dataframe(df):
    """Return a compact deterministic profile for a pandas DataFrame."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()
    missing = {str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": [str(c) for c in numeric],
        "categorical_columns": [str(c) for c in categorical],
        "numeric_summary": df[numeric].describe().round(6).to_dict() if numeric else {},
    }


# ------------------------------------------------------- uploaded datasets
#
# An uploaded dataset is an application-managed input, not a model-supplied
# path. The model never sees or chooses a filesystem location: it only ever
# calls load_uploaded_dataset() with zero arguments, which resolves to
# whatever file THIS chat currently owns. The on-disk path for a chat is
# always UPLOAD_ROOT/chat_<int chat_id>/dataset.<fmt> — chat_id is always an
# int from Telegram, and the Telegram-supplied file_name is used only for
# format sniffing and display, never as part of any filesystem path. This
# rules out path traversal / absolute-path escape by construction rather
# than by post-hoc validation.

SUPPORTED_UPLOAD_FORMATS = {
    "csv": {".csv"},
    "json": {".json"},
    "jsonl": {".jsonl", ".ndjson"},
}
SUPPORTED_UPLOAD_MIME_TYPES = {
    "text/csv": "csv", "application/csv": "csv",
    "application/json": "json",
    "application/x-ndjson": "jsonl", "application/jsonlines": "jsonl", "application/jsonl": "jsonl",
}


def _detect_upload_format(file_name: str, mime_type: str):
    """Return 'csv' / 'json' / 'jsonl', or None if unsupported/unrecognized."""
    name = (file_name or "").lower()
    for fmt, exts in SUPPORTED_UPLOAD_FORMATS.items():
        if any(name.endswith(ext) for ext in exts):
            return fmt
    return SUPPORTED_UPLOAD_MIME_TYPES.get((mime_type or "").lower())


def _chat_upload_dir(chat_id: int) -> str:
    # int(chat_id) can never contain "/", "..", or any path-control character,
    # so this cannot be used for traversal regardless of what a caller passes.
    return os.path.join(UPLOAD_ROOT, f"chat_{int(chat_id)}")


def _download_telegram_file(file_id: str, dest_path: str, max_bytes: int) -> None:
    """Download a Telegram-hosted file (by file_id) to dest_path, capping the
    streamed size. Raises on any failure. This only ever talks to Telegram's
    own fixed API host (same as tg()/TG_API elsewhere in this file) — it is
    not reachable from model-generated code and is unrelated to the SSRF
    guardrails that protect fetch_url/fetch_csv/etc. against arbitrary
    user-supplied URLs.
    """
    info = tg("getFile", file_id=file_id)
    if not info.get("ok"):
        raise RuntimeError(f"Telegram getFile failed: {info.get('description', 'unknown error')}")
    file_path = (info.get("result") or {}).get("file_path")
    if not file_path:
        raise RuntimeError("Telegram getFile response missing file_path")

    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    resp = requests.get(url, stream=True, timeout=60)
    try:
        resp.raise_for_status()
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"downloaded file exceeded {max_bytes}-byte limit while streaming")
                f.write(chunk)
    finally:
        resp.close()


def _set_dataset_context_message(chat_id: int, content: str) -> None:
    """Inject/replace the single 'current dataset' message in this chat's
    history, so subsequent turns see the new dataset's profile instead of a
    stale one from a previous upload."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history[:] = [
            m for m in history
            if not (isinstance(m.get("content"), str) and m["content"].startswith(DATASET_CONTEXT_TAG))
        ]
        history.append({"role": "user", "content": content})
        del history[:-20]


def handle_document_upload(chat_id: int, document: dict) -> str:
    """Download, validate, store, and profile an uploaded dataset for chat_id.
    Returns a JSON reply string in the same envelope shape as solve().
    On any failure, the previously-active dataset (if any) is left untouched.
    """
    file_name = document.get("file_name") or ""
    mime_type = document.get("mime_type") or ""
    file_id = document.get("file_id")
    declared_size = document.get("file_size")

    def fail(reason: str) -> str:
        log_event(event="upload_rejected", chat_id=chat_id, file_name=file_name, reason=reason)
        return json.dumps({"answer": f"upload rejected: {reason}", "log_url": LOG_URL}, ensure_ascii=False)

    if not file_id:
        return fail("missing file_id in Telegram document metadata")

    fmt = _detect_upload_format(file_name, mime_type)
    if fmt is None:
        return fail(f"unsupported file type (name={file_name!r}, mime={mime_type!r}); supported: csv, json, jsonl")

    if isinstance(declared_size, int) and declared_size > MAX_UPLOAD_BYTES:
        return fail(f"file too large ({declared_size} bytes; limit is {MAX_UPLOAD_BYTES} bytes)")

    chat_dir = _chat_upload_dir(chat_id)
    try:
        os.makedirs(chat_dir, exist_ok=True)
    except OSError as e:
        return fail(f"could not prepare storage: {e}")

    # Download/parse/profile into a STAGING file first. Nothing belonging to
    # a previously-active dataset is touched until every step below has
    # succeeded -- a failure at any point leaves the prior dataset exactly
    # as it was.
    staging_path = os.path.join(chat_dir, f".staging_{uuid4().hex}.{fmt}")

    log_event(event="upload_received", chat_id=chat_id, file_name=file_name,
              mime_type=mime_type, declared_size=declared_size, format=fmt)

    def _discard_staging():
        try:
            os.remove(staging_path)
        except OSError:
            pass

    try:
        _download_telegram_file(file_id, staging_path, MAX_UPLOAD_BYTES)
    except Exception as e:
        _discard_staging()
        return fail(f"download failed: {e}")

    try:
        if fmt == "csv":
            df = pd.read_csv(staging_path)
        elif fmt == "json":
            df = pd.read_json(staging_path)
        else:
            df = pd.read_json(staging_path, lines=True)
    except Exception as e:
        _discard_staging()
        return fail(f"could not parse file as {fmt}: {e}")

    try:
        profile = profile_dataframe(df)
    except Exception as e:
        _discard_staging()
        return fail(f"could not profile dataset: {e}")
    finally:
        del df  # don't hold a second full copy in memory once we have the profile

    # Validation fully passed — now, and only now, replace whatever this chat
    # had before (which may be a different format/extension) with the new
    # file, then promote the staging file to its final name.
    final_path = os.path.join(chat_dir, f"dataset.{fmt}")
    for existing_name in os.listdir(chat_dir):
        existing_path = os.path.join(chat_dir, existing_name)
        if existing_path != staging_path:
            try:
                os.remove(existing_path)
            except OSError:
                pass
    os.rename(staging_path, final_path)
    dest_path = final_path

    actual_size = os.path.getsize(dest_path)
    with _uploads_lock:
        _uploads[chat_id] = {
            "path": dest_path,
            "filename": file_name or f"dataset.{fmt}",
            "format": fmt,
            "profile": profile,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": actual_size,
        }

    profile_json = json.dumps(profile, ensure_ascii=False)[:3000]
    context_msg = (
        f"{DATASET_CONTEXT_TAG} A dataset was uploaded to this chat "
        f"(filename={file_name or ('dataset.' + fmt)!r}, format={fmt}, size_bytes={actual_size}). "
        f"Profile: {profile_json}. "
        "The file itself is on disk, not in this message. To analyze it, call run_python with "
        "code that calls load_uploaded_dataset() (no arguments) to get it as a pandas DataFrame. "
        "This replaces any dataset uploaded earlier in this chat."
    )
    _set_dataset_context_message(chat_id, context_msg)

    log_event(event="upload_stored", chat_id=chat_id, file_name=file_name, format=fmt,
              size_bytes=actual_size, rows=profile["shape"][0], columns=profile["shape"][1])

    ack = {
        "answer": (
            f"Dataset received ({file_name or fmt}, {profile['shape'][0]} rows x "
            f"{profile['shape'][1]} columns). Ask me anything about it."
        ),
        "log_url": LOG_URL,
    }
    return json.dumps(ack, ensure_ascii=False)


def _make_load_uploaded_dataset(chat_id: int):
    """Build a load_uploaded_dataset() closure bound to exactly one chat_id.

    The returned function takes NO arguments — there is no parameter through
    which model-generated code could ever supply a path. It always resolves
    to whatever file this specific chat currently owns, or raises a clear
    error if there isn't one.
    """
    def load_uploaded_dataset():
        with _uploads_lock:
            entry = _uploads.get(chat_id)
        if not entry:
            raise ValueError("no dataset has been uploaded in this chat yet")
        path, fmt = entry["path"], entry["format"]
        if not os.path.exists(path):
            raise ValueError("the uploaded dataset file is no longer available (it may have expired)")
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "json":
            return pd.read_json(path)
        return pd.read_json(path, lines=True)
    return load_uploaded_dataset


def _cleanup_expired_uploads() -> None:
    """Opportunistic sweep: delete per-chat upload directories that haven't
    been touched in UPLOAD_EXPIRY_SECONDS, and drop the matching in-memory
    entry so load_uploaded_dataset() fails cleanly instead of pointing at a
    half-deleted file. Called periodically from keepwarm_loop()."""
    if not os.path.isdir(UPLOAD_ROOT):
        return
    now = time.time()
    for entry_name in os.listdir(UPLOAD_ROOT):
        chat_dir = os.path.join(UPLOAD_ROOT, entry_name)
        if not os.path.isdir(chat_dir):
            continue
        try:
            age = now - os.path.getmtime(chat_dir)
            if age <= UPLOAD_EXPIRY_SECONDS:
                continue
            shutil.rmtree(chat_dir, ignore_errors=True)
            if entry_name.startswith("chat_"):
                try:
                    cid = int(entry_name[len("chat_"):])
                    with _uploads_lock:
                        _uploads.pop(cid, None)
                except ValueError:
                    pass
            log_event(event="upload_expired", chat_dir=entry_name, age_s=round(age, 1))
        except OSError as e:
            log_event(event="upload_cleanup_error", chat_dir=entry_name, error=str(e))


def _is_public_ip(ip_str: str) -> bool:
    """True only for addresses that are routable public internet addresses."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast \
            or ip.is_reserved or ip.is_unspecified:
        return False
    return True


def _validate_public_url(url: str) -> None:
    """Raise ValueError unless url is an http(s) URL that resolves only to
    public internet addresses. This blocks SSRF-style access to localhost,
    private LAN ranges, and link-local targets (which include the common
    cloud metadata address 169.254.169.254).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"blocked URL scheme: {parsed.scheme or '(none)'} (only http/https allowed)")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    if hostname.lower() in {"localhost"}:
        raise ValueError(f"blocked hostname: {hostname}")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve hostname: {hostname} ({e})")
    resolved_ips = {info[4][0] for info in infos}
    if not resolved_ips:
        raise ValueError(f"could not resolve hostname: {hostname}")
    for ip_str in resolved_ips:
        if not _is_public_ip(ip_str):
            raise ValueError(f"blocked target: {hostname} resolves to a non-public address ({ip_str})")


def _guarded_get(url: str, timeout: float = FETCH_TIMEOUT,
                  max_bytes: int = MAX_FETCH_BYTES, want_bytes: bool = False):
    """Fetch a public http(s) URL with SSRF/size/timeout/redirect guardrails.

    - Only http/https, only publicly-routable resolved addresses.
    - Redirects are followed manually (max MAX_FETCH_REDIRECTS), re-validating
      the target of every hop so a redirect cannot be used to reach a
      private/internal address.
    - Response size is capped both via Content-Length (if present) and by
      counting streamed bytes, so a server that lies about Content-Length
      can't exhaust memory.
    - Every attempt (success or failure) is written to the JSONL trace.
    """
    started = time.perf_counter()
    current_url = url
    resp = None
    try:
        for _ in range(MAX_FETCH_REDIRECTS + 1):
            _validate_public_url(current_url)
            resp = requests.get(
                current_url, timeout=timeout, stream=True, allow_redirects=False,
                headers={"User-Agent": "data-analyst-bot/1.0 (+public-data-fetch)"},
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                resp.close()
                if not location:
                    raise ValueError("redirect response had no Location header")
                current_url = requests.compat.urljoin(current_url, location)
                continue
            break
        else:
            raise ValueError(f"too many redirects (> {MAX_FETCH_REDIRECTS})")

        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            resp.close()
            raise ValueError(f"response too large ({content_length} bytes > {max_bytes}-byte limit)")

        chunks, total = [], 0
        for chunk in resp.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > max_bytes:
                resp.close()
                raise ValueError(f"response exceeded {max_bytes}-byte limit while streaming")
            chunks.append(chunk)
        raw = b"".join(chunks)

        log_event(event="web_fetch", url=current_url, status_code=resp.status_code,
                  bytes=total, latency_s=round(time.perf_counter() - started, 4))
        if want_bytes:
            return raw
        return raw.decode(resp.encoding or "utf-8", errors="replace")
    except Exception as e:
        log_event(event="web_fetch_error", url=current_url, error=str(e),
                  latency_s=round(time.perf_counter() - started, 4))
        raise
    finally:
        if resp is not None:
            resp.close()


def fetch_url(url: str, timeout: float = FETCH_TIMEOUT, max_bytes: int = MAX_FETCH_BYTES) -> str:
    """Fetch a public http(s) page/text and return it as a decoded string.
    Blocks private/internal targets; enforces a timeout and a size cap."""
    return _guarded_get(url, timeout=timeout, max_bytes=max_bytes, want_bytes=False)


def fetch_soup(url: str, timeout: float = FETCH_TIMEOUT, max_bytes: int = MAX_FETCH_BYTES,
               parser: str = "lxml") -> BeautifulSoup:
    """Fetch a public HTML page and return it already parsed with BeautifulSoup."""
    html = fetch_url(url, timeout=timeout, max_bytes=max_bytes)
    return BeautifulSoup(html, parser)


def fetch_table(url: str, table_index: int = 0, match=None,
                 timeout: float = FETCH_TIMEOUT, max_bytes: int = MAX_FETCH_BYTES):
    """Fetch a public HTML page and return one of its <table> elements as a DataFrame."""
    html = fetch_url(url, timeout=timeout, max_bytes=max_bytes)
    # flavor is pinned to "lxml" (a hard dependency, see requirements.txt).
    # Without this, pandas' flavor auto-detection can fall through to
    # html5lib on certain failures, and html5lib isn't installed — that
    # masks the real error (e.g. "no tables found") behind an unrelated
    # ImportError.
    kwargs = {"flavor": "lxml"}
    if match:
        kwargs["match"] = match
    tables = pd.read_html(io.StringIO(html), **kwargs)
    if not tables:
        raise ValueError("no <table> elements found on the page")
    if table_index >= len(tables):
        raise ValueError(f"table_index {table_index} out of range ({len(tables)} tables found)")
    return tables[table_index]


def fetch_csv(url: str, timeout: float = FETCH_TIMEOUT, max_bytes: int = MAX_FETCH_BYTES, **read_csv_kwargs):
    """Fetch a public CSV file and return it as a DataFrame."""
    text = fetch_url(url, timeout=timeout, max_bytes=max_bytes)
    return pd.read_csv(io.StringIO(text), **read_csv_kwargs)


def fetch_excel(url: str, timeout: float = FETCH_TIMEOUT, max_bytes: int = MAX_FETCH_BYTES, **read_excel_kwargs):
    """Fetch a public Excel file and return it as a DataFrame."""
    raw = _guarded_get(url, timeout=timeout, max_bytes=max_bytes, want_bytes=True)
    return pd.read_excel(io.BytesIO(raw), **read_excel_kwargs)


def run_python(code: str, chat_id: int) -> str:
    """Validate and execute analysis code, returning captured stdout/errors.

    Any outcome that should trigger bounded recovery in solve() is prefixed
    with "ERROR:" or "GUARDRAIL_ERROR:" — this includes real exceptions raised
    during exec(), not just guardrail rejections and timeouts.

    chat_id scopes load_uploaded_dataset() to exactly this chat's own upload.
    """
    valid, reason = validate_python(code)
    if not valid:
        return f"GUARDRAIL_ERROR: {reason}"

    out = io.StringIO()
    result: dict = {"ok": None}

    def target():
        env = {
            "__name__": "__main__",
            "profile_dataframe": profile_dataframe,
            "fetch_url": fetch_url,
            "fetch_soup": fetch_soup,
            "fetch_table": fetch_table,
            "fetch_csv": fetch_csv,
            "fetch_excel": fetch_excel,
            "load_uploaded_dataset": _make_load_uploaded_dataset(chat_id),
        }
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        log_event(event="tool_timeout", timeout_s=PY_TIMEOUT)
        # Known limitation: this thread is daemonized but not forcibly killed;
        # it may continue running in the background after we give up on it.
        return f"ERROR: code timed out after {PY_TIMEOUT}s"

    text = out.getvalue()
    if result["ok"] is False:
        tail = text[-MAX_OUTPUT_CHARS:] if text else "(no output before the exception)"
        return f"ERROR: exception during execution:\n{tail}"
    return text[-MAX_OUTPUT_CHARS:] if text else "(no output — use print())"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, bs4, openpyxl are installed. Raw network access "
                "(requests, urllib) is NOT permitted. To retrieve public web data, "
                "use the pre-provided functions instead — no import needed: "
                "fetch_url(url) -> str (raw text), "
                "fetch_soup(url) -> BeautifulSoup, "
                "fetch_table(url, table_index=0) -> DataFrame (from an HTML <table>), "
                "fetch_csv(url, **kwargs) -> DataFrame, "
                "fetch_excel(url, **kwargs) -> DataFrame. "
                "Only public http/https URLs are allowed (internal/private addresses "
                "are blocked); responses are capped at "
                f"{MAX_FETCH_BYTES} bytes and {FETCH_TIMEOUT}s. "
                "If the user has uploaded a dataset to this chat (a message will say so), "
                "call load_uploaded_dataset() — no arguments — to get it as a DataFrame; "
                "there is no way to point it at any other file. "
                "Always print() what you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks.
2. The message may embed data inline, or reference a public dataset (MOSPI, data.gov.in, etc.). Use the run_python tool to fetch data and compute — do not guess numeric results you can compute. Retrieve public web data ONLY via the pre-provided fetch_url/fetch_soup/fetch_table/fetch_csv/fetch_excel functions (no import needed, raw requests/urllib are blocked). pd.read_csv/pd.read_json only work when called directly on an inline io.StringIO(...)/BytesIO(...) buffer (e.g. pd.read_csv(io.StringIO(text))) — not on a URL, a file path, or a variable holding a buffer built on an earlier line; use fetch_csv/fetch_excel for anything remote. For well-known published statistics (e.g. "which state has the highest maternal mortality rate per MOSPI/SRS"), you may answer from reliable knowledge if fetching fails.
3. The message usually spells out the exact JSON shape it wants, e.g. Reply with ONLY {"answer": {"state": "<state>"}, "log_url": "..."}.
4. When you are ready to answer, reply with ONLY that JSON object — no prose, no markdown fences. Use a placeholder like "LOG_URL" for the log_url value; the harness substitutes the real URL. Match the requested shape for "answer" EXACTLY (keys, nesting, types: numbers as numbers unless a string is asked for).
5. If the message does not specify a shape, reply {"answer": <your concise answer>, "log_url": "LOG_URL"}.
6. If a mid-conversation message is only setup/context ("I will send data next"), still reply with {"answer": "ok", "log_url": "LOG_URL"} unless it asks something.
7. Round numbers as instructed; if unspecified, give reasonable precision. Never add keys that were not asked for inside "answer".
"""


# ---------------------------------------------------------------- llm
def chat_completion(messages, use_tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    if use_tools:
        body["tools"] = TOOLS
    r = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (data-analyst-bot)",
        },
        json=body,
        timeout=180,
    )
    
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    run_started = time.perf_counter()
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)
    final_text = None
    deadline = time.time() + ANSWER_BUDGET
    recovery_attempts = 0
    tool_calls_count = 0

    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append({"role": "user", "content": "Time is up. Reply NOW with only your best final JSON object."})
        try:
            llm_started = time.perf_counter()
            msg = chat_completion(messages, use_tools=not out_of_time)
            llm_latency = time.perf_counter() - llm_started
            log_event(event="llm_response", chat_id=chat_id, step=step, latency_s=round(llm_latency, 4))
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, step=step, error=str(e))
            time.sleep(2)
            try:
                llm_started = time.perf_counter()
                msg = chat_completion(messages, use_tools=True)
                log_event(event="llm_retry", chat_id=chat_id, step=step, latency_s=round(time.perf_counter()-llm_started, 4))
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                tool_calls_count += 1
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError:
                    code = tc["function"]["arguments"]
                log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                tool_started = time.perf_counter()
                output = run_python(code, chat_id)
                tool_latency = time.perf_counter() - tool_started
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000], latency_s=round(tool_latency, 4))
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})

                if output.startswith("GUARDRAIL_ERROR:") or output.startswith("ERROR:"):
                    if recovery_attempts < MAX_RECOVERY_ATTEMPTS:
                        recovery_attempts += 1
                        messages.append({
                            "role": "user",
                            "content": (
                                "The previous Python tool call failed. Fix the code and retry. "
                                f"This is recovery attempt {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}. "
                                "Do not repeat blocked operations; use an allowed data-analysis approach."
                            ),
                        })
                        log_event(event="recovery_attempt", chat_id=chat_id, step=step, attempt=recovery_attempts)
            continue

        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) if final_text else None
    if obj is None:
        obj = {"answer": (final_text or "unable to determine").strip()[:1000]}
    if "answer" not in obj:
        obj = {"answer": obj}
    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    total_latency = time.perf_counter() - run_started
    log_event(
        event="answer", chat_id=chat_id, reply=reply,
        total_latency_s=round(total_latency, 4),
        tool_calls=tool_calls_count,
        recovery_attempts=recovery_attempts,
    )
    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)

    print("SEND STATUS:", r.status_code)
    print("SEND BODY:", r.text)

    return r.json()


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return

    chat_id = msg["chat"]["id"]
    document = msg.get("document")
    text = msg.get("text") or msg.get("caption") or ""

    if not document and not text:
        return

    try:
        with _get_chat_lock(chat_id):
            if document:
                upload_reply = handle_document_upload(chat_id, document)
                # If the upload came with a question (caption) or the message
                # also has plain text, answer it now — the dataset's profile
                # is already in this chat's history by this point. Otherwise
                # just send the deterministic upload acknowledgment.
                reply = solve(chat_id, text) if text else upload_reply
            else:
                reply = solve(chat_id, text)
    except Exception:
        print(traceback.format_exc())      # <-- ADD THIS
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})

    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            r = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            )

            

            resp = r.json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down. Also sweeps
    expired uploaded-dataset directories on the same cadence, so we don't
    need a separate cleanup thread/scheduler."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass
        try:
            _cleanup_expired_uploads()
        except Exception as e:
            log_event(event="upload_cleanup_error", error=str(e))


# ---------------------------------------------------------------- web app
app = FastAPI()


@app.on_event("startup")
def _start():
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}


@app.post("/debug/ask")
def debug_ask(payload: dict, x_eval_token: str = Header(default="")):
    """Direct agent-invocation endpoint for the evaluation harness.

    Disabled entirely unless EVAL_TOKEN is set on the server AND the caller
    supplies a matching X-Eval-Token header. This exists so evaluation does
    not require driving the bot through Telegram, which is not automatable.
    """
    if not EVAL_TOKEN or x_eval_token != EVAL_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")
    question = (payload or {}).get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="'question' is required")
    chat_id = (payload or {}).get("chat_id", -1)
    with _get_chat_lock(chat_id):
        reply = solve(chat_id, question)
    return {"reply": reply}
