"""Low-latency streaming TTS pipeline with ordered emission."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from src.core.models.stream import ChatStream
from src.kernel.logger import Logger

from .tts import TTSArtifact, TTSBackend, TTSRequest


@dataclass(slots=True)
class _PipelineResult:
    artifact: TTSArtifact | None = None
    error: Exception | None = None


class StreamingTTSPipeline:
    """Synthesize in parallel, but emit strictly in submit order."""

    def __init__(
        self,
        *,
        backend: TTSBackend,
        chat_stream: ChatStream,
        max_parallel: int,
        logger: Logger | None,
        empty_audio_retry_count: int = 0,
    ) -> None:
        self._backend = backend
        self._chat_stream = chat_stream
        self._logger = logger
        self._empty_audio_retry_count = max(0, int(empty_audio_retry_count))
        self._semaphore = asyncio.Semaphore(max(1, int(max_parallel or 1)))

        self._next_submit_seq = 0
        self._next_emit_seq = 0
        self._closed = False
        self._jobs_ready = asyncio.Event()
        self._result_futures: dict[int, asyncio.Future[_PipelineResult]] = {}
        self._on_emit_callbacks: dict[int, Callable[[], None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._emit_task: asyncio.Task[None] | None = None
        self.emit_success_count = 0

    async def submit_text(
        self,
        text: str,
        *,
        emotion: str | None = None,
        provider: str | None = None,
        markers: dict[str, object] | None = None,
        on_emit: Callable[[], None] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("StreamingTTSPipeline is already closed.")

        sentence = text.strip()
        if not sentence:
            return

        seq = self._next_submit_seq
        self._next_submit_seq += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_PipelineResult] = loop.create_future()
        self._result_futures[seq] = future
        if on_emit is not None:
            self._on_emit_callbacks[seq] = on_emit
        self._jobs_ready.set()

        if self._emit_task is None:
            self._emit_task = asyncio.create_task(self._emit_loop())

        task = asyncio.create_task(
            self._synthesize(
                seq=seq,
                text=sentence,
                emotion=emotion,
                provider=provider,
                markers=markers,
                future=future,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        self._closed = True
        self._jobs_ready.set()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._emit_task is not None:
            await self._emit_task
            self._emit_task = None

    async def _synthesize(
        self,
        *,
        seq: int,
        text: str,
        emotion: str | None,
        provider: str | None,
        markers: dict[str, object] | None,
        future: asyncio.Future[_PipelineResult],
    ) -> None:
        try:
            async with self._semaphore:
                request = TTSRequest(
                    stream_id=self._chat_stream.stream_id,
                    text=text,
                    emotion=emotion,
                    provider=provider,
                    markers=dict(markers or {}),
                )
                artifact = await self._backend.synthesize(request)
                artifact = await self._retry_empty_audio(request=request, artifact=artifact)
        except Exception as exc:  # noqa: BLE001
            if self._logger is not None:
                self._logger.warning(
                    f"Streaming TTS synthesis failed for seq={seq}: {exc}"
                )
            if not future.done():
                future.set_result(_PipelineResult(error=exc))
            return

        if not future.done():
            future.set_result(_PipelineResult(artifact=artifact))

    async def _retry_empty_audio(self, *, request: TTSRequest, artifact: TTSArtifact) -> TTSArtifact:
        current = artifact
        for _ in range(self._empty_audio_retry_count):
            if current.audio:
                return current
            current = await self._backend.synthesize(request)
        return current

    async def _emit_loop(self) -> None:
        while True:
            future = self._result_futures.get(self._next_emit_seq)
            if future is None:
                if self._closed and self._next_emit_seq >= self._next_submit_seq:
                    break
                self._jobs_ready.clear()
                # Re-check after clear to avoid lost wakeup:
                # submit_text may have set the event between our .get() and .clear().
                if self._next_emit_seq in self._result_futures:
                    continue
                await self._jobs_ready.wait()
                continue

            result = await future
            self._result_futures.pop(self._next_emit_seq, None)
            self._next_emit_seq += 1

            artifact = result.artifact
            if result.error is not None or artifact is None:
                continue
            if not artifact.audio:
                if self._logger is not None:
                    self._logger.warning(
                        "Streaming TTS produced empty audio; skipping emission."
                    )
                continue
            try:
                await self._backend.emit(artifact, self._chat_stream)
                self.emit_success_count += 1
                on_emit = self._on_emit_callbacks.pop(self._next_emit_seq - 1, None)
                if on_emit is not None:
                    on_emit()
            except Exception as exc:  # noqa: BLE001
                if self._logger is not None:
                    self._logger.warning(
                        f"Streaming TTS emit failed for seq={self._next_emit_seq - 1}: {exc}"
                    )


__all__ = ["StreamingTTSPipeline"]
