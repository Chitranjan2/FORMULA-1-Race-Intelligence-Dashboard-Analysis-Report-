"""Cached, retrying JSON fetch layer shared by both API clients.

Design notes:
    * Responses are memoised on disk forever — race data is immutable, so a
      re-run costs no network at all.
    * ``Accept-Encoding: gzip`` is requested explicitly. OpenF1 serves multi-MB
      lap payloads, and uncompressed transfer was the dominant cost of a full
      ingest.
    * OpenF1 throttles aggressively and returns plain ``429`` without
      rate-limit headers, so a 429 triggers an exponential backoff rather than
      a fast retry loop.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, HTTP_TIMEOUT, MAX_RETRIES, REQUEST_PAUSE, USER_AGENT

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _cache_file(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _decode(response) -> Any:
    raw = response.read()
    if (response.headers.get("Content-Encoding") or "").lower() == "gzip":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def fetch_json(url: str, *, refresh: bool = False, timeout: int = HTTP_TIMEOUT) -> Any:
    """Fetch ``url`` as JSON, memoising the response on disk."""
    cache = _cache_file(url)
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = _decode(response)
            cache.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(REQUEST_PAUSE)
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRY_STATUS:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after and str(retry_after).isdigit() else min(2 ** attempt, 30)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts") from last_error
