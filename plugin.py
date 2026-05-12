"""sherpa-onnx ASR 实时语音通话专用 Chatter。"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, AsyncGenerator

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base import BaseChatter, BasePlugin, Failure, Success, Wait, WaitResumeEvent
from src.core.components.base.action import BaseAction
from src.core.components.loader import register_plugin
from src.core.components.types import ChatType
from src.core.config import get_core_config
from src.core.models.stream import ChatStream
from src.core.prompt import get_prompt_manager
from src.kernel.llm import LLMPayload, ROLE, Text, ToolRegistry
from src.kernel.llm.payload.tooling import LLMUsable

from .config import SherpaOnnxVoiceChatterConfig
from .markers import parse_speech_segments
from .prompt_builder import SYSTEM_PROMPT, USER_PROMPT, VoiceChatterPromptBuilder
from .runner import run_voice_conversation
from .tts import build_tts_backend, synthesize_segments


logger = get_logger("voice_chatter")

_PASS_AND_WAIT = "action-pass_and_wait"


class SayAction(BaseAction):
    """把要说的话发送到 TTS 后端并交给适配器播放。"""

    action_name = "say"
    action_description = (
        "在实时语音通话中说出一段话。content 会进入 TTS 后端并由适配器播放。"
        "支持 [wait:1] 控制下一段播放前等待 1 秒，支持 [emotion:happy]...[/emotion] 标记情绪。"
        "[wait] 只影响语音片段播放间隔，不会让聊天流等待；说完等待用户时请另外调用 pass_and_wait。"
    )
    chatter_allow = ["voice_chatter"]
    associated_platforms = ["local_asr"]
    dependencies = ["asr_adapter:adapter:asr_adapter"]

    async def execute(
        self,
        content: Annotated[str, "要通过 TTS 说出的内容，可包含 [wait:n] 和 [emotion:name] 标记"],
    ) -> tuple[bool, str]:
        """执行语音播放动作。"""

        plugin_config = getattr(self.plugin, "config", None)
        split_enabled = True
        max_parallel = 4
        empty_audio_retry_count = 1
        if isinstance(plugin_config, SherpaOnnxVoiceChatterConfig):
            split_enabled = bool(plugin_config.tts.sentence_split_enabled)
            max_parallel = int(plugin_config.tts.max_parallel_segments)
            empty_audio_retry_count = int(plugin_config.tts.empty_audio_retry_count)

        segments = parse_speech_segments(content or "", split_sentences=split_enabled)
        if not segments:
            return True, "没有可播放的语音内容"

        backend = build_tts_backend(plugin_config, logger)
        artifacts = await synthesize_segments(
            backend=backend,
            stream_id=self.chat_stream.stream_id,
            segments=segments,
            max_parallel=max_parallel,
            empty_audio_retry_count=empty_audio_retry_count,
        )

        success_count = 0
        failed_reasons: list[str] = []
        for segment, artifact in zip(segments, artifacts, strict=False):
            error = artifact.metadata.get("error") if isinstance(artifact.metadata, dict) else None
            if error:
                failed_reasons.append(str(error))
                logger.error(f"TTS 合成失败，跳过播放: {segment.text} ({error})")
                continue
            if not artifact.audio:
                failed_reasons.append("TTS 后端未返回音频数据")
                logger.error(f"TTS 后端未返回音频数据，跳过播放: {segment.text}")
                continue
            if segment.wait_before > 0:
                await asyncio.sleep(segment.wait_before)
            if await backend.emit(artifact, self.chat_stream):
                success_count += 1

        if success_count == 0 and failed_reasons:
            return False, f"TTS 合成失败: {failed_reasons[0]}"
        return True, f"已提交 {success_count}/{len(segments)} 段语音到适配器播放"


class VoicePassAndWaitAction(BaseAction):
    """等待用户继续语音输入或等待指定秒数后主动恢复。"""

    action_name = "pass_and_wait"
    action_description = (
        "为实时语音通话登记等待点。说完话后调用它等待用户继续说话；"
        "seconds 为空时等待新语音输入，传入秒数时到时主动恢复。"
    )
    chatter_allow = ["voice_chatter"]
    associated_platforms = ["local_asr"]

    async def execute(
        self,
        seconds: Annotated[float | None, "等待秒数；为空则等待新的用户语音输入"] = None,
    ) -> tuple[bool, str]:
        """登记等待状态。"""

        if seconds is None:
            return True, "已登记等待新的用户语音输入"
        return True, f"已登记等待 {seconds} 秒后继续语音通话"


class SherpaOnnxVoiceChatter(BaseChatter):
    """sherpa-onnx ASR 实时语音通话专用 Chatter。"""

    chatter_name = "voice_chatter"
    chatter_description = "sherpa-onnx ASR 实时语音通话专用 Chatter"
    associated_platforms = ["local_asr"]
    chat_type = ChatType.PRIVATE
    dependencies = ["asr_adapter:adapter:asr_adapter"]
    stream_tick_interval = 0.1
    allow_message_buffer = False

    def _get_plugin_config(self) -> SherpaOnnxVoiceChatterConfig | None:
        """返回插件配置。"""

        config = getattr(self.plugin, "config", None)
        return config if isinstance(config, SherpaOnnxVoiceChatterConfig) else None

    def apply_stream_runtime_options(self, chat_stream: Any) -> None:
        """把语音通话的流运行时配置写入当前 stream。"""

        plugin_config = self._get_plugin_config()
        if plugin_config is not None:
            self.stream_tick_interval = float(plugin_config.plugin.tick_interval)
            self.allow_message_buffer = bool(plugin_config.plugin.allow_message_buffer)
        super().apply_stream_runtime_options(chat_stream)

    async def _build_system_prompt(self, chat_stream: ChatStream) -> str:
        """构建语音通话系统提示词。"""

        return await VoiceChatterPromptBuilder.build_system_prompt(
            self._get_plugin_config(),
            chat_stream,
        )

    def _build_history_text(self, chat_stream: ChatStream) -> str:
        """构建历史消息文本。"""

        return VoiceChatterPromptBuilder.build_history_text(chat_stream, self.format_message_line)

    async def _build_user_prompt(
        self,
        chat_stream: ChatStream,
        history_text: str,
        unread_lines: str,
        extra: str = "",
    ) -> str:
        """构建语音通话用户提示词。"""

        return await VoiceChatterPromptBuilder.build_user_prompt(
            chat_stream,
            history_text,
            unread_lines,
            extra,
        )

    @staticmethod
    def _build_negative_behaviors_extra() -> str:
        """构建行为提醒。"""

        return VoiceChatterPromptBuilder.build_negative_behaviors_extra()

    def _is_action_suspend_enabled(self) -> bool:
        """读取纯 Action 回合的挂起开关。"""

        plugin_config = self._get_plugin_config()
        return plugin_config is None or bool(plugin_config.plugin.enable_action_suspend)

    @staticmethod
    def _append_user_payload(response: Any, text: str) -> None:
        """向当前 LLM 上下文追加 USER 文本。"""

        response.add_payload(LLMPayload(ROLE.USER, Text(text)))

    async def inject_usables(self, request: Any) -> ToolRegistry:
        """注入语音 Chatter 可用工具，排除 stop/send_text/sub-agent 管理工具。"""

        usables = await self.get_llm_usables()
        usables = await self.modify_llm_usables(usables)
        blocked_names = {
            "action-send_text",
            "action-stop_conversation",
            "create_agent",
            "get_agent",
            "kill_agent",
        }

        registry = ToolRegistry()
        for usable in usables:
            schema = usable.to_schema()
            name = str(schema.get("function", {}).get("name", ""))
            if name in blocked_names:
                continue
            registry.register(usable)

        if registry.get_all():
            request.add_payload(LLMPayload(ROLE.TOOL, registry.get_all()))  # type: ignore[arg-type]
        return registry

    async def execute(self) -> AsyncGenerator[Wait | Success | Failure, WaitResumeEvent | None]:
        """执行语音 Chatter 主循环。"""

        from src.core.managers.stream_manager import get_stream_manager

        stream_manager = get_stream_manager()
        chat_stream = await stream_manager.activate_stream(self.stream_id)
        if chat_stream is None:
            logger.error(f"无法激活聊天流: {self.stream_id}")
            yield Failure("无法激活聊天流")
            return

        self.apply_stream_runtime_options(chat_stream)
        plugin_config = self._get_plugin_config()
        retry_limit = 1 if plugin_config is None else int(plugin_config.plugin.plain_text_retry_limit)

        runner = run_voice_conversation(
            chatter=self,
            chat_stream=chat_stream,
            logger=logger,
            pass_call_name=_PASS_AND_WAIT,
            plain_text_retry_limit=max(0, retry_limit),
            enable_action_suspend=self._is_action_suspend_enabled(),
        )
        resume_event: WaitResumeEvent | None = None
        while True:
            try:
                result = await runner.asend(resume_event)
            except StopAsyncIteration:
                return
            resume_event = yield result


@register_plugin
class SherpaOnnxVoiceChatterPlugin(BasePlugin):
    """sherpa-onnx ASR 实时语音 Chatter 插件。"""

    plugin_name = "voice_chatter"
    plugin_version = "1.0.0"
    plugin_description = "sherpa-onnx ASR 实时语音通话专用 Chatter"
    configs = [SherpaOnnxVoiceChatterConfig]
    dependent_components = ["asr_adapter:adapter:asr_adapter"]

    async def on_plugin_loaded(self) -> None:
        """注册语音 Chatter 提示词模板。"""

        from src.core.prompt import min_len, optional, wrap

        personality = get_core_config().personality
        get_prompt_manager().get_or_create(
            name="voice_chatter_system_prompt",
            template=SYSTEM_PROMPT,
            policies={
                "nickname": optional(personality.nickname),
                "alias_names": optional("、".join(personality.alias_names)),
                "personality_core": optional(personality.personality_core),
                "personality_side": optional(personality.personality_side),
                "identity": optional(personality.identity),
                "reply_style": optional(personality.reply_style),
                "background_story": optional(personality.background_story)
                .then(min_len(10))
                .then(wrap("# 背景故事\n", "\n")),
                "safety_guidelines": optional("\n".join(personality.safety_guidelines)),
                "negative_behaviors": optional("\n".join(personality.negative_behaviors)),
                "voice_guide": optional(""),
            },
        )
        get_prompt_manager().get_or_create(
            name="voice_chatter_user_prompt",
            template=USER_PROMPT,
            policies={
                "stream_name": optional("未知通话"),
                "current_time": optional("未知时间"),
                "platform": optional("local_asr"),
                "history": optional("").then(min_len(2)).then(wrap("# 历史通话内容\n", "\n")),
                "unreads": optional("").then(min_len(2)).then(wrap("# 新识别到的语音\n", "\n")),
                "extra": optional("").then(min_len(2)).then(wrap("# 额外提醒\n", "\n")),
            },
        )

    def get_components(self) -> list[type]:
        """返回插件组件。"""

        return [SherpaOnnxVoiceChatter, SayAction, VoicePassAndWaitAction]


__all__ = [
    "SayAction",
    "SherpaOnnxVoiceChatter",
    "SherpaOnnxVoiceChatterPlugin",
    "VoicePassAndWaitAction",
]
