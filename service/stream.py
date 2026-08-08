"""Server-Sent Events plumbing for streamed LLM answers.

Why SSE and not WebSocket
-------------------------
The payload here is strictly server -> client token push. SSE is plain HTTP, so it
traverses ordinary proxies and needs no upgrade handshake; the protocol already
defines resumption (``id:`` lines plus the ``Last-Event-ID`` request header) and
browser-side auto-reconnect, whereas WebSocket would require hand-rolled heartbeat
and reconnect logic for a channel the client never writes to.

Resumption semantics
--------------------
An LLM completion cannot be re-generated deterministically, so "resume" here means
*replay what was already produced*: each stream is buffered under a stream id, and
a reconnect carrying ``Last-Event-ID`` replays the deltas after that sequence
number instead of starting a fresh (and different) completion. Buffers are bounded
and LRU-evicted, so this is short-window recovery for a dropped connection, not a
durable log.

Threading
---------
``LLMClient.chat_stream()`` is a *synchronous* generator (the OpenAI SDK's blocking
client). Iterating it directly inside an async handler would block the event loop
and serialise every concurrent request, so it runs on a worker thread and chunks
reach the loop through an ``asyncio.Queue`` via ``call_soon_threadsafe``.

Event protocol
--------------
    event: meta   data: {"stream_id": ..., "request_id": ...}    # once, first
    event: delta  data: {"text": "..."}          id: <seq>       # many
    event: done   data: {"text": "<full>", "ttft_ms": ..., "elapsed_ms": ...}
    event: error  data: {"error": "..."}
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator

from service.observability import LLM_ERRORS, LLM_TTFT

log = logging.getLogger("medigraph.stream")

#: Sentinel pushed onto the bridge queue when the producer thread is finished.
_DONE = object()

#: How many completed streams stay replayable, and how much text each may hold.
MAX_BUFFERED_STREAMS = 64
MAX_BUFFERED_CHARS = 64_000

#: Emitted when the model is slow so proxies and browsers keep the connection open.
HEARTBEAT_SECONDS = 15.0


def sse_frame(event: str, data: dict, event_id: int | None = None) -> str:
    """Encode one SSE frame.

    ``json.dumps`` guarantees a single-line payload, which matters because a raw
    newline inside ``data:`` would be parsed as a field break by the SSE spec.
    """
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


def sse_comment(text: str = "keep-alive") -> str:
    """A comment frame: ignored by clients, but resets idle timers on the path."""
    return f": {text}\n\n"


class StreamBuffer:
    """Ordered deltas for one stream, so a reconnect can replay the tail."""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.deltas: list[str] = []
        self.chars = 0
        self.truncated = False
        self.finished = False
        self.error: str | None = None
        self.created = time.time()

    def append(self, piece: str) -> int:
        """Store a delta and return its 1-based sequence number."""
        if self.chars + len(piece) > MAX_BUFFERED_CHARS:
            # Stop retaining once over budget; the live stream is unaffected, only
            # replay coverage is lost, which the resume response reports.
            self.truncated = True
        else:
            self.chars += len(piece)
        self.deltas.append(piece)
        return len(self.deltas)

    @property
    def text(self) -> str:
        return "".join(self.deltas)

    def after(self, seq: int) -> list[tuple[int, str]]:
        """Deltas strictly after `seq`, paired with their sequence numbers."""
        return [(i, piece) for i, piece in enumerate(self.deltas, start=1) if i > seq]


class StreamRegistry:
    """Bounded LRU of recent streams, keyed by stream id."""

    def __init__(self, max_streams: int = MAX_BUFFERED_STREAMS):
        self._streams: OrderedDict[str, StreamBuffer] = OrderedDict()
        self._max = max_streams
        self._lock = threading.Lock()

    def create(self) -> StreamBuffer:
        buffer = StreamBuffer(uuid.uuid4().hex[:16])
        with self._lock:
            self._streams[buffer.stream_id] = buffer
            while len(self._streams) > self._max:
                self._streams.popitem(last=False)
        return buffer

    def get(self, stream_id: str) -> StreamBuffer | None:
        with self._lock:
            buffer = self._streams.get(stream_id)
            if buffer is not None:
                self._streams.move_to_end(stream_id)
            return buffer


registry = StreamRegistry()


def parse_last_event_id(raw: str | None) -> int:
    """Parse the `Last-Event-ID` header into a sequence number (0 when absent)."""
    if not raw:
        return 0
    try:
        return max(0, int(raw.strip()))
    except (TypeError, ValueError):
        return 0


#: A producer receives an `emit` callback and blocks until finished. Returning a
#: value is allowed and captured (the QA route returns its full evidence payload).
Producer = Callable[[Callable[[str], None]], object]


def iterator_producer(make_iterator: Callable[[], Iterator[str]]) -> Producer:
    """Adapt a plain sync generator to the callback-style `Producer` interface."""

    def produce(emit: Callable[[str], None]) -> None:
        for piece in make_iterator():
            emit(piece)

    return produce


async def bridge_producer(
    produce: Producer,
    result: dict,
    queue_size: int = 256,
) -> AsyncIterator[str | BaseException]:
    """Run a blocking producer on one worker thread, yielding deltas on the loop.

    Yields the exception object rather than raising it, so the caller can turn a
    provider failure into an SSE ``error`` frame instead of tearing down the
    response mid-body -- by then a 200 status and headers have already been sent.

    The producer's return value is stored in `result["value"]`.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

    def emit(piece: str) -> None:
        # Bounded queue + blocking put gives back-pressure: when the client reads
        # slower than the model emits, this thread waits instead of buffering the
        # whole completion in memory.
        asyncio.run_coroutine_threadsafe(queue.put(piece), loop).result()

    def worker() -> None:
        try:
            result["value"] = produce(emit)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop).result()

    thread = threading.Thread(target=worker, name="llm-stream", daemon=True)
    thread.start()
    while True:
        item = await queue.get()
        if item is _DONE:
            return
        yield item


async def sse_llm_stream(
    produce: Producer,
    request_id: str,
    last_event_id: int = 0,
    stream_id: str | None = None,
    done_extra: Callable[[object], dict] | None = None,
) -> AsyncIterator[str]:
    """Full SSE response body for one streamed completion.

    When `stream_id` names a buffered stream and `last_event_id` is set, the tail is
    replayed instead of issuing a new completion.

    `done_extra` maps the producer's return value into extra fields on the terminal
    ``done`` frame -- the QA route uses it to ship evidence and citations alongside
    the streamed prose.
    """
    # ---- resume path: replay, do not re-generate ------------------------- #
    if stream_id:
        buffered = registry.get(stream_id)
        if buffered is not None:
            yield sse_frame(
                "meta",
                {
                    "stream_id": stream_id,
                    "request_id": request_id,
                    "resumed": True,
                    "from_seq": last_event_id,
                    "replay_truncated": buffered.truncated,
                },
            )
            for seq, piece in buffered.after(last_event_id):
                yield sse_frame("delta", {"text": piece}, event_id=seq)
            if buffered.finished:
                if buffered.error:
                    yield sse_frame("error", {"error": buffered.error})
                else:
                    yield sse_frame("done", {"text": buffered.text, "resumed": True})
                return
            # Still in flight: the live producer owns the tail, so stop here and let
            # the client reconnect rather than attaching two consumers to one stream.
            yield sse_frame("done", {"text": buffered.text, "partial": True})
            return

    # ---- live path -------------------------------------------------------- #
    buffer = registry.create()
    yield sse_frame(
        "meta", {"stream_id": buffer.stream_id, "request_id": request_id, "resumed": False}
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    last_flush = time.perf_counter()
    result: dict = {}

    try:
        async for item in bridge_producer(produce, result):
            if isinstance(item, BaseException):
                buffer.error = str(item)
                buffer.finished = True
                LLM_ERRORS.inc()
                log.warning("stream failed: %s", item)
                yield sse_frame("error", {"error": str(item)})
                return
            if first_token_at is None:
                first_token_at = time.perf_counter()
                LLM_TTFT.observe(first_token_at - started)
            seq = buffer.append(item)
            yield sse_frame("delta", {"text": item}, event_id=seq)
            now = time.perf_counter()
            if now - last_flush > HEARTBEAT_SECONDS:
                last_flush = now
                yield sse_comment()
    except asyncio.CancelledError:
        # Client disconnected. Keep the buffer so a reconnect can resume, then let
        # cancellation propagate so the server does not treat this as success.
        buffer.finished = True
        log.info("client disconnected: stream_id=%s", buffer.stream_id)
        raise

    buffer.finished = True
    elapsed = time.perf_counter() - started
    ttft = (first_token_at - started) if first_token_at else elapsed
    payload = {
        "text": buffer.text,
        "stream_id": buffer.stream_id,
        "ttft_ms": round(ttft * 1000, 2),
        "elapsed_ms": round(elapsed * 1000, 2),
        "deltas": len(buffer.deltas),
    }
    if done_extra is not None:
        try:
            payload.update(done_extra(result.get("value")))
        except Exception:  # noqa: BLE001 - never lose the answer over a metadata bug
            log.exception("done_extra failed for stream_id=%s", buffer.stream_id)
    yield sse_frame("done", payload)


#: Headers required for SSE to survive intermediaries.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Disable proxy buffering (nginx), otherwise frames are held back and the
    # whole point of streaming -- early first byte -- is lost.
    "X-Accel-Buffering": "no",
}
