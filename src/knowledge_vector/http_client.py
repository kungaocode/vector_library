from __future__ import annotations

import json
import time
from http.client import IncompleteRead, RemoteDisconnected
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def post_json(
    url: str, payload: dict[str, Any], *, api_key: str = "", timeout_s: float = 60,
) -> Any:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Model API endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("Model API URL must not contain credentials")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    limit = 16 * 1024 * 1024
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                raw = response.read(limit + 1)
            break
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            if api_key:
                detail = detail.replace(api_key, "[redacted]")
            raise RuntimeError(f"Model API returned HTTP {exc.code}: {detail}") from None
        except (IncompleteRead, RemoteDisconnected, URLError, TimeoutError, OSError) as exc:
            if attempt < 2:
                time.sleep(attempt + 1)
                continue
            detail = str(exc).replace(api_key, "[redacted]") if api_key else str(exc)
            raise RuntimeError(f"Model API request failed after 3 attempts: {detail}") from None
    if len(raw) > limit:
        raise RuntimeError("Model API response exceeded the 16 MiB limit")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("Model API returned invalid JSON") from None
