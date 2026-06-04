"""Configuration for the voice_chatter plugin."""

from __future__ import annotations

from typing import ClassVar

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class VoiceChatterConfig(BaseConfig):
    """Runtime configuration for voice_chatter."""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "Voice chatter configuration"

    @config_section("plugin", title="Plugin", tag="plugin")
    class PluginSection(SectionBase):
        enabled: bool = Field(default=True, description="Enable voice_chatter")
        tick_interval: float = Field(default=1.0, description="Tick interval for the chat stream")
        allow_message_buffer: bool = Field(
            default=False,
            description="Allow message buffering on the stream",
        )
        plain_text_retry_limit: int = Field(
            default=1,
            description="Retry count when the model answers with plain text instead of say action",
        )
        enable_action_suspend: bool = Field(
            default=True,
            description=(
                "Enable suspend behavior for pure-action turns. When disabled, pure action "
                "results continue follow-up like normal tool results."
            ),
        )

    @config_section("tts", title="TTS", tag="tts")
    class TTSSection(SectionBase):
        endpoint: str = Field(
            default="http://127.0.0.1:8000/router/tts_http_server/api/tts/v1/synthesize",
            description="TTS HTTP synthesize endpoint",
        )
        timeout: float = Field(default=30.0, description="TTS HTTP request timeout")
        max_parallel_segments: int = Field(
            default=4,
            description="Max parallel synthesized segments",
        )
        empty_audio_retry_count: int = Field(
            default=1,
            description="Retry count when TTS returns empty audio",
        )
        sentence_split_enabled: bool = Field(
            default=True,
            description="Split text into sentences before TTS",
        )
        mime_type: str = Field(default="audio/wav", description="TTS audio MIME type")
        provider: str = Field(
            default="qwen_tts",
            description="TTS provider name; empty means use server default",
        )
        emit_text_on_tts_failure: bool = Field(
            default=False,
            description="Fallback to sending text when TTS fails",
        )

    @config_section("low_latency_streaming", title="Streaming", tag="low_latency_streaming")
    class LowLatencyStreamingSection(SectionBase):
        enabled: bool = Field(default=False, description="Enable low-latency streaming TTS")
        max_parallel_tts: int = Field(
            default=2,
            description="Max parallel TTS tasks in streaming mode",
        )
        min_sentence_chars: int = Field(
            default=4,
            description="Minimum sentence length before submitting streaming TTS",
        )
        continuation_grace_ms: int = Field(
            default=80,
            description="Lookahead window for continuation punctuation before flushing a sentence",
        )
        flush_tail_on_done: bool = Field(
            default=True,
            description="Flush incomplete tail text when the tool-call stream ends",
        )
        require_native_tool_calling: bool = Field(
            default=True,
            description="Require native tool calling support; tool_call_compat falls back automatically",
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    low_latency_streaming: LowLatencyStreamingSection = Field(
        default_factory=LowLatencyStreamingSection
    )
