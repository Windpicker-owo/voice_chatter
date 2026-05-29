"""语音 Chatter 配置。"""

from __future__ import annotations

from typing import ClassVar

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class VoiceChatterConfig(BaseConfig):
    """语音 Chatter 配置。"""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "语音 Chatter 配置"

    @config_section("plugin", title="插件设置", tag="plugin")
    class PluginSection(SectionBase):
        """插件基础配置。"""

        enabled: bool = Field(default=True, description="是否启用语音 Chatter")
        tick_interval: float = Field(default=1.0, description="该语音聊天流的 Tick 间隔")
        allow_message_buffer: bool = Field(default=False, description="是否允许消息缓冲")
        plain_text_retry_limit: int = Field(default=1, description="模型返回纯文本时的提醒重试次数")
        enable_action_suspend: bool = Field(
            default=True,
            description="是否启用纯 Action 回合的挂起机制。关闭后，纯 Action 结果会像常规工具结果一样继续 follow-up，而不是立即等待用户。",
        )

    @config_section("tts", title="TTS 设置", tag="tts")
    class TTSSection(SectionBase):
        """TTS 后端配置。"""

        endpoint: str = Field(
            default="http://127.0.0.1:8000/router/tts_http_server/api/tts/v1/synthesize",
            description="TTS HTTP 合成接口地址",
        )
        timeout: float = Field(default=30.0, description="TTS HTTP 请求超时时间")
        max_parallel_segments: int = Field(default=4, description="最大并行合成句子数")
        empty_audio_retry_count: int = Field(default=1, description="TTS 返回空音频时的重试次数")
        sentence_split_enabled: bool = Field(default=True, description="是否按句切分并并行合成")
        mime_type: str = Field(default="audio/wav", description="TTS 音频 MIME 类型")
        provider: str = Field(default="qwen_tts", description="TTS provider 名称，留空则使用服务端默认 provider")
        emit_text_on_tts_failure: bool = Field(default=False, description="TTS 失败时是否回退发送文本")

    @config_section("low_latency_streaming", title="Streaming", tag="low_latency_streaming")
    class LowLatencyStreamingSection(SectionBase):
        """低延迟流式 TTS 设置。"""

        enabled: bool = Field(default=False, description="是否启用低延迟流式 TTS")
        max_parallel_tts: int = Field(default=2, description="流式模式下的最大并行 TTS 任务数")
        min_sentence_chars: int = Field(default=4, description="提交流式 TTS 前的最小句子长度")
        flush_tail_on_done: bool = Field(default=True, description="工具调用流结束时是否刷新未完成的文本尾部")
        require_native_tool_calling: bool = Field(
            default=True,
            description="是否需要原生工具调用支持；tool_call_compat 会自动回退",
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    low_latency_streaming: LowLatencyStreamingSection = Field(
        default_factory=LowLatencyStreamingSection
    )
