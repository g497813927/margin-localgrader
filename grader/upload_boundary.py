"""Bound request bodies before Starlette can parse or spool multipart uploads."""

from __future__ import annotations

from starlette.formparsers import MultiPartParser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_REQUEST_BYTES = 16 * 1024 * 1024


class BodySizeLimitMiddleware:
    """Buffer at most 16 MiB in memory, then replay a complete request body.

    The 16 MiB envelope includes multipart metadata around the endpoint's 15 MiB
    file limit. Oversized bodies never reach the multipart parser, even when the
    client omits Content-Length or streams the body in chunks. No partial body is
    forwarded on overflow or disconnect.

    Starlette's normal UploadFile parser uses SpooledTemporaryFile. Its process-
    wide spool threshold is raised to this same envelope, so accepted files do
    not roll over to disk. Keep upload handling to read/seek/close; calling a
    spooled file's fileno() explicitly would force a disk rollover.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int = MAX_REQUEST_BYTES):
        if max_body_bytes <= 0:
            raise ValueError("The request size limit must be positive.")
        self.app = app
        self.max_body_bytes = max_body_bytes
        # The complete HTTP body is capped before the parser is entered, making
        # this a bounded memory setting rather than an unlimited upload spool.
        MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, max_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        lengths = [value for name, value in scope.get("headers", []) if name.lower() == b"content-length"]
        if lengths:
            try:
                declared = [int(value) for value in lengths]
                if any(value < 0 for value in declared) or len(set(declared)) != 1:
                    raise ValueError
            except ValueError:
                await self._reject(send, 400, b'{"detail":"Invalid request size."}')
                return
            if declared[0] > self.max_body_bytes:
                await self._too_large(send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                body.clear()
                return
            if message["type"] != "http.request":
                body.clear()
                await self._reject(send, 400, b'{"detail":"Invalid request body."}')
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                body.clear()
                await self._too_large(send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        complete_body = bytes(body)
        body.clear()
        delivered = False

        async def replay() -> Message:
            nonlocal complete_body, delivered
            if not delivered:
                delivered = True
                message: Message = {"type": "http.request", "body": complete_body, "more_body": False}
                complete_body = b""
                return message
            return await receive()

        try:
            await self.app(scope, replay, send)
        finally:
            complete_body = b""

    async def _too_large(self, send: Send) -> None:
        await self._reject(send, 413, b'{"detail":"The request is too large. Upload files must be at most 15 MiB; total requests must be at most 16 MiB."}')

    @staticmethod
    async def _reject(send: Send, status: int, body: bytes) -> None:
        await send({
            "type": "http.response.start", "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"x-content-type-options", b"nosniff"),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})
