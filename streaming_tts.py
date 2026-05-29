"""Low-latency streaming TTS pipeline with ordered emission."""

from __future__ import annotations

import asyncio
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
        self._tasks: set[asyncio.Task[None]] = set()
        self._emit_task: asyncio.Task[None] | None = None

    async def submit_text(self, text: str) -> None:
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
        self._jobs_ready.set()

        if self._emit_task is None:
            self._emit_task = asyncio.create_task(self._emit_loop())

        task = asyncio.create_task(self._synthesize(seq=seq, text=sentence, future=future))
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
        future: asyncio.Future[_PipelineResult],
    ) -> None:
        try:
            async with self._semaphore:
                artifact = await self._backend.synthesize(
                    TTSRequest(
                        stream_id=self._chat_stream.stream_id,
                        text=text,
                    )
                )
                artifact = await self._retry_empty_audio(text=text, artifact=artifact)
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

    async def _retry_empty_audio(self, *, text: str, artifact: TTSArtifact) -> TTSArtifact:
        current = artifact
        for _ in range(self._empty_audio_retry_count):
            if current.audio:
                return current
            current = await self._backend.synthesize(
                TTSRequest(
                    stream_id=self._chat_stream.stream_id,
                    text=text,
                )
            )
        return current

    async def _emit_loop(self) -> None:
        while True:
            future = self._result_futures.get(self._next_emit_seq)
            if future is None:
                if self._closed and self._next_emit_seq >= self._next_submit_seq:
                    break
                self._jobs_ready.clear()
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
            except Exception as exc:  # noqa: BLE001
                if self._logger is not None:
                    self._logger.warning(
                        f"Streaming TTS emit failed for seq={self._next_emit_seq - 1}: {exc}"
                    )


__all__ = ["StreamingTTSPipeline"]
