# voice_chatter

## 概述

`voice_chatter` 是面向实时语音通话场景的专用 Chatter。

它依赖本地 ASR 适配器接收用户语音输入，再通过 TTS HTTP 后端把回复内容合成为语音并回送给适配器播放。和普通文本 Chatter 相比，它的核心目标不是多轮文字对话，而是“听到用户说话后，生成分段语音回复”。

## 提供的组件

- `voice_chatter:chatter:voice_chatter`
- `voice_chatter:action:say`
- `voice_chatter:action:pass_and_wait`

其中：

- `voice_chatter` 是实时语音通话主控制器
- `say` 用于把一段文本送进 TTS 后端并交由适配器播放
- `pass_and_wait` 用于在说完后等待用户继续说话，或等待指定秒数后恢复

## 依赖

插件依赖：

- `asr_adapter`
- `tts_http_server`

关键运行时关系：

- 输入来自 `asr_adapter:adapter:asr_adapter`
- TTS 输出默认通过 `tts_http_server` 的 HTTP 接口完成

## 配置

配置文件路径：

- `config/plugins/voice_chatter/config.toml`

主要配置节：

- `plugin`：启用状态、tick 间隔、消息缓冲、纯文本提醒重试次数
- `tts`：TTS 合成接口地址、超时、并行句子数、空音频重试、默认 provider、失败时是否回退文本

默认 TTS 接口地址是：

- `http://127.0.0.1:8000/router/tts_http_server/api/tts/v1/synthesize`

## 工作方式

1. 语音输入先由 `asr_adapter` 转成统一消息流。
2. `voice_chatter` 在 `local_asr` 平台上运行语音对话控制逻辑。
3. `say` action 会解析 `[wait:n]`、`[emotion:xxx]...[/emotion]` 等标记，并按句切分。
4. 切分后的语音段通过 HTTP TTS 后端并行合成。
5. 合成结果被重新包装成 voice 消息，由适配器负责顺序播放。

## 使用建议

- 该插件适合一问一答、陪伴式语音对话、实时角色语音交互等场景。
- 如果你只需要文字聊天，不应使用此插件。
- 如果你替换 TTS 后端，实现兼容 `tts_http_server` 协议的 provider 即可，不必改本插件。

## 相关插件

- `plugins/asr_adapter`
- `plugins/tts_http_server`
- `plugins/qwen_tts_provider`
