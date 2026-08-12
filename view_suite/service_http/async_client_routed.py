# async_client_routed_hrw.py
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .multipart import decode_multipart
from .async_client import AsyncUnifiedClient

LOGGER = logging.getLogger(__name__)


class HRWRoutedAsyncUnifiedClient(AsyncUnifiedClient):
    """
    AsyncUnifiedClient + HRW (Rendezvous hashing) scene routing + failover-after-N-failures.

    Key properties:
      - Deterministic across processes/machines: uses md5(scene_id|url) scoring.
      - For each scene_id, compute a ranked list of candidate URLs (best -> worst).
      - Retry policy:
          * attempts 0..F use best URL
          * attempt F+1 uses second best
          * attempt F+2 uses third best
          * ...
      - Backoff/jitter behavior preserved: delay = backoff * 2**attempt * jitter
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

        # ---- Routing controls ----
        route_by_scene: bool = True,
        failover_after_failures: int = 0,
        promote_on_success: bool = True,  # pin to the url that succeeded (per-client-instance)
        cache_ranked_candidates: bool = True,  # cache HRW ranking per scene_id (per-client-instance)
    ):
        super().__init__(
            base_url,
            token=token,
            timeout=timeout,
            max_connections=max_connections,
            log_retries=log_retries,
            retry_log_level=retry_log_level,
        )

        self.route_by_scene = bool(route_by_scene)
        self.failover_after_failures = max(0, int(failover_after_failures))
        self.promote_on_success = bool(promote_on_success)
        self.cache_ranked_candidates = bool(cache_ranked_candidates)

        self._route_lock = asyncio.Lock()
        self._scene_to_primary_index: Dict[str, int] = {}
        self._scene_to_ranked_indices: Dict[str, List[int]] = {}

    @staticmethod
    def _md5_u64(text: str) -> int:
        d = hashlib.md5(text.encode("utf-8")).digest()
        return int.from_bytes(d[:8], byteorder="big", signed=False)

    def _rank_urls_hrw(self, scene_id: str) -> List[int]:
        """
        HRW ranking: score(url) = md5(scene_id + '|' + url)
        Return indices sorted by descending score.
        """
        scores = []
        for i, u in enumerate(self.base_urls):
            s = self._md5_u64(f"{scene_id}|{u}")
            scores.append((s, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [i for _, i in scores]

    async def _get_ranked_indices(self, scene_id: Optional[str]) -> List[int]:
        k = len(self.base_urls)
        if k <= 1:
            return [0]
        if not self.route_by_scene or not scene_id:
            return list(range(k))  # deterministic fallback order

        async with self._route_lock:
            if self.cache_ranked_candidates and scene_id in self._scene_to_ranked_indices:
                return self._scene_to_ranked_indices[scene_id]

            ranked = self._rank_urls_hrw(scene_id)

            # If we've previously "promoted" a primary, make sure it is ranked first
            # (still keep the rest of ranking to have deterministic failover order).
            primary = self._scene_to_primary_index.get(scene_id)
            if primary is not None and primary in ranked:
                ranked.remove(primary)
                ranked.insert(0, primary)

            if self.cache_ranked_candidates:
                self._scene_to_ranked_indices[scene_id] = ranked
            return ranked

    async def _promote_primary(self, scene_id: Optional[str], url_index: int) -> None:
        if not self.promote_on_success or not self.route_by_scene or not scene_id:
            return
        async with self._route_lock:
            self._scene_to_primary_index[scene_id] = url_index
            # If ranking cached, move promoted index to front
            if scene_id in self._scene_to_ranked_indices:
                ranked = self._scene_to_ranked_indices[scene_id]
                if url_index in ranked:
                    ranked.remove(url_index)
                ranked.insert(0, url_index)
                self._scene_to_ranked_indices[scene_id] = ranked

    @staticmethod
    def _pick_candidate_index(ranked: List[int], attempt: int, failover_after_failures: int) -> int:
        """
        attempts 0..F -> ranked[0]
        attempt F+1 -> ranked[1]
        attempt F+2 -> ranked[2]
        ...
        If attempt exceeds candidates, continue cycling from the end in a stable way.
        """
        if not ranked:
            return 0
        if attempt <= failover_after_failures:
            return ranked[0]
        offset = attempt - failover_after_failures
        pos = min(offset, len(ranked) - 1)
        return ranked[pos]

    async def render(
        self,
        meta: Optional[Dict[str, Any]] = None,
        images: Optional[List[Image.Image]] = None,
        *,
        retries: int = 4,
        backoff: float = 0.5,
    ) -> Tuple[Dict[str, Any], List[Image.Image]]:
        params = {"token": self.token} if self.token else None

        scene_id = None
        if isinstance(meta, dict):
            scene_id = meta.get("scene_id")

        # Build multipart form fields
        data: Dict[str, str] = {}
        if meta is not None:
            data["meta"] = json.dumps(meta, ensure_ascii=False)

        files = []
        buffers: List[io.BytesIO] = []
        for i, img in enumerate(images or []):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            buffers.append(buf)  # keep alive
            files.append(("images", (f"{i}.png", buf, "image/png")))

        ranked = await self._get_ranked_indices(scene_id)

        last_exc: Optional[BaseException] = None

        for attempt in range(int(retries) + 1):
            url_index = self._pick_candidate_index(ranked, attempt, self.failover_after_failures)
            current_base_url = self.base_urls[url_index]
            url = f"{current_base_url}/render"

            try:
                r = await self._client.post(
                    url,
                    params=params,
                    data=data,
                    files=files if files else None,
                    timeout=self.timeout,
                )

                # Treat 503 as retriable busy signal
                if r.status_code == 503:
                    raise RuntimeError("server busy")

                r.raise_for_status()

                ctype = r.headers.get("content-type", "")
                if not ctype.lower().startswith("multipart/"):
                    raise RuntimeError(f"Unexpected response Content-Type: {ctype}")

                # If we succeeded on a non-first candidate, optionally promote for future calls (within this process)
                if scene_id and ranked and url_index != ranked[0]:
                    await self._promote_primary(scene_id, url_index)
                    # refresh local ranked ordering (promoted becomes first)
                    ranked = await self._get_ranked_indices(scene_id)

                return decode_multipart(ctype, r.content)

            except Exception as e:
                last_exc = e

                if attempt == retries:
                    LOGGER.error(
                        "[HRWRoutedAsyncUnifiedClient] request failed (scene_id=%s, url=%s) after %d retries; error=%s",
                        scene_id,
                        current_base_url,
                        retries,
                        repr(e),
                    )
                    raise

                # Preserve original backoff semantics
                delay = min(float(backoff) * (2 ** attempt), 8.0) * (0.7 + 0.6 * random.random())

                if self.log_retries:
                    next_attempt = attempt + 1
                    next_idx = self._pick_candidate_index(ranked, next_attempt, self.failover_after_failures)
                    next_url = self.base_urls[next_idx]

                    LOGGER.log(
                        self.retry_log_level,
                        "[HRWRoutedAsyncUnifiedClient] retrying (scene_id=%s, current_url=%s, next_url=%s, attempt=%d/%d, delay=%.2fs, error_type=%s, error=%s)",
                        scene_id,
                        current_base_url,
                        next_url,
                        next_attempt,
                        retries,
                        delay,
                        type(e).__name__,
                        repr(e),
                    )

                await asyncio.sleep(delay)

        raise last_exc or RuntimeError("Unknown error")
