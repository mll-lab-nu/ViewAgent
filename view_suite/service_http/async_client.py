# async_client.py
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
import httpcore
from PIL import Image

from .multipart import decode_multipart

LOGGER = logging.getLogger(__name__)


class AsyncUnifiedClient:
    """
    Asynchronous HTTP client for the unified render service.

    This client:
      - Sends multipart/form-data requests with:
          * 'meta' JSON string field (optional)
          * 'images' repeated PNG files (optional)
      - Expects multipart/mixed responses and decodes them into:
          * metadata dict
          * list of PIL Images
      - Retries transient failures with exponential backoff + jitter.
      - Logs retry attempts at INFO/WARNING level for observability.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: float = 120.0,
        max_connections: int = 100,
        log_retries: bool = True,
        retry_log_level: int = logging.WARNING,
    ):
        """
        Args:
            base_url:
                Service base URL, e.g. "http://localhost:8765".
                Can also be multiple URLs separated by semicolons, e.g. "http://url1;http://url2".
                When multiple URLs are provided, they will be used in round-robin fashion on retries.
                The client will call POST {base_url}/render.

            token:
                Optional API token for authentication.
                Sent as query parameter '?token=...'.

            timeout:
                Per-request timeout in seconds.
                This bounds how long the client will wait for:
                  - connection establishment
                  - request upload
                  - server processing
                  - full response download

            max_connections:
                Maximum number of concurrent HTTP connections in the pool.
                For stress tests, increase this to avoid client-side bottlenecks.

            log_retries:
                If True, emit log lines for retry attempts.

            retry_log_level:
                Logging level used for all retry logs. WARNING is a good default.
        """
        # Parse base_url: support multiple URLs separated by semicolons
        self.base_urls = [url.strip().rstrip("/") for url in base_url.split(";")]
        self.base_url = self.base_urls[0]  # Keep backward compatibility

        self.token = token
        self.timeout = float(timeout)

        self.log_retries = bool(log_retries)
        self.retry_log_level = int(retry_log_level)

        limits = httpx.Limits(max_connections=int(max_connections))
        # A remote render service may be served over HTTPS with a self-signed
        # certificate. Verification stays on by default; set RENDER_TLS_NO_VERIFY=1 to
        # accept a self-signed pair, or leave it on and point SSL_CERT_FILE at the cert.
        verify = os.getenv("RENDER_TLS_NO_VERIFY", "0").strip().lower() not in (
            "1", "true", "yes",
        )
        self._client = httpx.AsyncClient(
            timeout=self.timeout, limits=limits, verify=verify
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release pooled connections."""
        await self._client.aclose()

    async def render(
        self,
        meta: Optional[Dict[str, Any]] = None,
        images: Optional[List[Image.Image]] = None,
        *,
        retries: int = 8,
        backoff: float = 2.0,
    ) -> Tuple[Dict[str, Any], List[Image.Image]]:
        """
        Send a render request.

        Args:
            meta:
                Optional metadata dict for the service.
                Will be JSON-serialized into the form field named 'meta'.

            images:
                Optional input images.
                Each image is encoded as PNG and uploaded as a repeated form field
                named 'images'.

            retries:
                Number of retries after the initial attempt.
                Total attempts = retries + 1.

            backoff:
                Initial backoff delay (seconds) for exponential backoff.
                Retry delay grows as:
                    delay = backoff * 2^attempt * jitter
                where jitter is a random multiplier in [0.7, 1.3).

        Returns:
            (response_meta, response_images)

        Raises:
            httpx.HTTPError:
                For non-retriable HTTP errors after r.raise_for_status().

            RuntimeError:
                For server busy (503) or unexpected response Content-Type.
        """
        params = {"token": self.token} if self.token else None

        # Optional convenience for logs
        scene_id = None
        if isinstance(meta, dict):
            scene_id = meta.get("scene_id")

        # Build multipart/form-data fields
        data: Dict[str, str] = {}
        if meta is not None:
            data["meta"] = json.dumps(meta, ensure_ascii=False)

        # Build multipart file fields (keep buffers alive during request)
        files = []
        buffers: List[io.BytesIO] = []
        for i, img in enumerate(images or []):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            buffers.append(buf)  # Prevent GC while the request is in-flight
            files.append(("images", (f"{i}.png", buf, "image/png")))

        last_exc: Optional[BaseException] = None

        for attempt in range(int(retries) + 1):
            # Round-robin across multiple base URLs on retries
            current_base_url = self.base_urls[attempt % len(self.base_urls)]
            url = f"{current_base_url}/render"

            try:
                r = await self._client.post(
                    url,
                    params=params,
                    data=data,
                    files=files if files else None,
                    timeout=self.timeout,
                )

                # Treat 503 as retriable "server busy"
                if r.status_code == 503:
                    raise RuntimeError("server busy")

                r.raise_for_status()

                ctype = r.headers.get("content-type", "")
                if not ctype.lower().startswith("multipart/"):
                    raise RuntimeError(f"Unexpected response Content-Type: {ctype}")

                return decode_multipart(ctype, r.content)

            except Exception as e:
                last_exc = e

                # If out of retries, log and re-raise
                if attempt == retries:
                    LOGGER.error(
                        "[AsyncUnifiedClient] request failed (scene_id=%s, url=%s) after %d retries; error=%s",
                        scene_id,
                        current_base_url,
                        retries,
                        repr(e),
                    )
                    raise

                # Exponential backoff with jitter to avoid synchronized retry storms
                delay = float(backoff) * (2 ** attempt) * (0.7 + 0.6 * random.random())

                # Determine next URL for retry
                next_url_index = (attempt + 1) % len(self.base_urls)
                next_base_url = self.base_urls[next_url_index]

                if self.log_retries:
                    LOGGER.log(
                        self.retry_log_level,
                        "[AsyncUnifiedClient] retrying request (scene_id=%s, current_url=%s, next_url=%s, attempt=%d/%d, delay=%.2fs, error_type=%s, error=%s)",
                        scene_id,
                        current_base_url,
                        next_base_url,
                        attempt + 1,       # retry count (1-based)
                        retries,
                        delay,
                        type(e).__name__,
                        repr(e),
                    )

                await asyncio.sleep(delay)

        # Should not happen, but keep a safe fallback.
        raise last_exc or RuntimeError("Unknown error")
