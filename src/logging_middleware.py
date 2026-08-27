"""ASGI middleware that logs every request received and response sent."""
from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("ingestion_pipeline_service.http")

_MAX_LOG_BODY_CHARS = 2000


def _truncate(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    return text if len(text) <= _MAX_LOG_BODY_CHARS else text[:_MAX_LOG_BODY_CHARS] + "...<truncated>"


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method/path/body on the way in and status/body on the way out."""

    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        logger.info(
            "--> %s %s body=%s", request.method, request.url.path, _truncate(
                body)
        )

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk
        logger.info(
            "<-- %s %s status=%s body=%s (%.1fms)",
            request.method, request.url.path, response.status_code,
            _truncate(resp_body), duration_ms,
        )
        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
