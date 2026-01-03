import asyncio
import base64
from collections import deque
import json
import logging
import os
import re
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
    InvokeModelWithResponseStreamInput,
    ResourceNotFoundException,
    ThrottlingException,
    ValidationException,
)
from aws_sdk_bedrock_runtime.config import Config, SigV4AuthScheme
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_core.interceptors import Interceptor
from smithy_core.interfaces import TypedProperties
from smithy_http import Field

# ログ設定
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# AWS SDK のログを抑制（冗長すぎるため）
logging.getLogger("aws_sdk_bedrock_runtime").setLevel(logging.WARNING)
logging.getLogger("awscrt").setLevel(logging.WARNING)
logging.getLogger("smithy_aws_core").setLevel(logging.WARNING)
logging.getLogger("smithy_core").setLevel(logging.WARNING)
logging.getLogger("smithy_http").setLevel(logging.WARNING)
logging.getLogger("smithy_aws_event_stream").setLevel(logging.WARNING)
logging.getLogger("smithy_json").setLevel(logging.WARNING)

# すべての AWS 関連ライブラリのログレベルを一括で上げる
for name in logging.root.manager.loggerDict:
    if any(x in name.lower() for x in ['aws', 'smithy', 'boto']):
        logging.getLogger(name).setLevel(logging.WARNING)

app = FastAPI()

MODEL_ID = "amazon.nova-2-sonic-v1:0"
TRANSLATION_MODEL_ID_DEFAULT = "anthropic.claude-haiku-4-5-20251001-v1:0"
TRANSLATION_MODEL_ID = os.getenv("TRANSLATION_MODEL_ID", TRANSLATION_MODEL_ID_DEFAULT)
TRANSLATION_MODEL_ID_FALLBACK_DEFAULT = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
TRANSLATION_MODEL_ID_FALLBACK = os.getenv(
    "TRANSLATION_MODEL_ID_FALLBACK", TRANSLATION_MODEL_ID_FALLBACK_DEFAULT
)
TRANSLATION_MAX_TOKENS = int(os.getenv("TRANSLATION_MAX_TOKENS", "400"))
TRANSLATION_DEBOUNCE_SECONDS = float(os.getenv("TRANSLATION_DEBOUNCE_SECONDS", "0.4"))
TRANSLATION_MIN_INTERVAL_SECONDS = float(os.getenv("TRANSLATION_MIN_INTERVAL_SECONDS", "0.6"))
TRANSLATION_MAX_BATCH_CHARS = int(os.getenv("TRANSLATION_MAX_BATCH_CHARS", "1200"))

CATCHUP_MODEL_ID_DEFAULT = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CATCHUP_MODEL_ID = os.getenv("CATCHUP_MODEL_ID", CATCHUP_MODEL_ID_DEFAULT)
CATCHUP_MODEL_ID_FALLBACK = os.getenv("CATCHUP_MODEL_ID_FALLBACK", CATCHUP_MODEL_ID_DEFAULT)
CATCHUP_MAX_TOKENS = int(os.getenv("CATCHUP_MAX_TOKENS", "350"))
CATCHUP_MIN_INTERVAL_SECONDS = float(os.getenv("CATCHUP_MIN_INTERVAL_SECONDS", "15"))
CATCHUP_LOG_MAX_ITEMS = int(os.getenv("CATCHUP_LOG_MAX_ITEMS", "200"))
CATCHUP_LOG_MAX_SECONDS = int(os.getenv("CATCHUP_LOG_MAX_SECONDS", "1800"))
CATCHUP_MAX_INPUT_CHARS = int(os.getenv("CATCHUP_MAX_INPUT_CHARS", "6000"))
AWS_REGION_DEFAULT = "ap-northeast-1"

# Bedrock側ストリーム寿命(8分)手前で更新
RENEW_SECONDS = 7 * 60 + 45  # 7m45s
# プロキシタイムアウト回避用キープアライブ間隔
KEEPALIVE_SECONDS = 30


class SigV4AuthSchemeWithChecksum(SigV4AuthScheme):
    def signer_properties(self, *, context: TypedProperties) -> dict:
        props = super().signer_properties(context=context)
        props["content_checksum_enabled"] = True
        return props


class FixDuplicateContentTypeInterceptor(Interceptor):
    def modify_before_signing(self, context):
        req = context.transport_request
        try:
            content_type = req.fields["content-type"]
        except Exception:
            return req

        if not getattr(content_type, "values", None) or len(content_type.values) <= 1:
            return req

        preferred = None
        for val in content_type.values:
            if val.lower().startswith("application/json"):
                preferred = val
                break
        if preferred is None:
            preferred = content_type.values[0]

        req.fields.set_field(Field(name=content_type.name, values=[preferred]))
        return req


def get_bedrock_client(region: str) -> BedrockRuntimeClient:
    cfg = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    cfg.set_auth_scheme(SigV4AuthSchemeWithChecksum(service="bedrock"))
    cfg.interceptors.append(FixDuplicateContentTypeInterceptor())
    return BedrockRuntimeClient(config=cfg)


class NovaSonicSession:
    def __init__(self, client: BedrockRuntimeClient, model_id: str):
        self.client = client
        self.model_id = model_id

        self.prompt_name = str(uuid.uuid4())
        self.system_content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self.stream = None
        self.is_active = False

        self.current_role: Optional[str] = None
        self.current_generation_stage: Optional[str] = None

    async def open(self, *, system_prompt: str) -> None:
        logger.info("Opening Bedrock session...")
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        logger.info("✓ Bedrock session opened successfully")

        await self._send_event(json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.0,
                    },
                    "turnDetectionConfiguration": {"endpointingSensitivity": "HIGH"},
                }
            }
        }))

        await self._send_event(json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 8000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        }))

        # SYSTEM text
        await self._send_event(json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.system_content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }))
        await self._send_event(json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.system_content_name,
                    "content": system_prompt,
                }
            }
        }))
        await self._send_event(json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.system_content_name,
                }
            }
        }))

        # USER audio container
        await self._send_event(json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }))
        logger.info("✓ Bedrock session initialized and ready")

    async def send_audio(self, pcm16le_bytes: bytes) -> None:
        if not self.is_active:
            return
        b64 = base64.b64encode(pcm16le_bytes).decode("utf-8")
        await self._send_event(json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": b64,
                }
            }
        }))

    async def recv_once(self) -> Optional[dict]:
        if not self.is_active or self.stream is None:
            return None
        output = await self.stream.await_output()
        result = await output[1].receive()
        if not result.value or not result.value.bytes_:
            return None
        try:
            return json.loads(result.value.bytes_.decode("utf-8"))
        except Exception:
            return None

    async def close(self) -> None:
        if not self.is_active or self.stream is None:
            return
        self.is_active = False
        logger.info("Closing Bedrock session...")
        try:
            await self._send_event(json.dumps({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                    }
                }
            }))
            await self._send_event(json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}}))
            await self._send_event(json.dumps({"event": {"sessionEnd": {}}}))
        finally:
            try:
                await self.stream.input_stream.close()
            except Exception:
                pass
            logger.info("✓ Bedrock session closed")

    async def _send_event(self, event_json: str) -> None:
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(chunk)


def _extract_generation_stage(content_start_event: dict) -> Optional[str]:
    amf = content_start_event.get("additionalModelFields")
    if not amf:
        return None
    try:
        return json.loads(amf).get("generationStage")
    except Exception:
        return None


_INTERRUPTED_TAG_RE = re.compile(r"^\s*\{\s*[\"']interrupted[\"']\s*:\s*true\s*\}\s*$", re.I)


def _is_interrupted_tag(text: str) -> bool:
    if not text:
        return False
    if _INTERRUPTED_TAG_RE.match(text):
        return True
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        obj = json.loads(stripped)
    except Exception:
        return False
    return isinstance(obj, dict) and obj.get("interrupted") is True and len(obj) == 1


def _extract_response_stream_bytes(event: object) -> Optional[bytes]:
    if event is None:
        return None

    if isinstance(event, dict):
        for key in ("output", "chunk", "value"):
            part = event.get(key)
            if isinstance(part, (bytes, bytearray)):
                return bytes(part)
            if isinstance(part, dict):
                for bytes_key in ("bytes", "bytes_"):
                    b = part.get(bytes_key)
                    if isinstance(b, (bytes, bytearray)):
                        return bytes(b)
            else:
                b = getattr(part, "bytes_", None)
                if isinstance(b, (bytes, bytearray)):
                    return bytes(b)
        return None

    for attr in ("output", "chunk", "value"):
        part = getattr(event, attr, None)
        if part is None:
            continue
        b = getattr(part, "bytes_", None)
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)

    b = getattr(event, "bytes_", None)
    if isinstance(b, (bytes, bytearray)):
        return bytes(b)
    return None


async def stream_prompt_text_deltas(
    client: BedrockRuntimeClient,
    *,
    prompt: str,
    model_id: str,
) -> object:
    """
    Claude 4.5 Haiku をストリーミング呼び出しし、テキスト delta を yield する。
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": TRANSLATION_MAX_TOKENS,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    req = InvokeModelWithResponseStreamInput(
        model_id=model_id,
        content_type="application/json",
        accept="application/json",
        body=json.dumps(body).encode("utf-8"),
    )

    async with (await client.invoke_model_with_response_stream(req)) as stream:
        async for event in stream.output_stream:
            payload = _extract_response_stream_bytes(event)
            if not payload:
                continue
            try:
                chunk = json.loads(payload.decode("utf-8"))
            except Exception:
                continue

            if chunk.get("type") != "content_block_delta":
                continue
            delta = chunk.get("delta") or {}
            delta_text = delta.get("text")
            if not delta_text:
                continue
            yield delta_text


_INDEXED_LINE_RE = re.compile(r"^\s*(\d{1,6})\s*[\t:]\s*(.*?)\s*$")


def _parse_indexed_output_line(line: str) -> Optional[tuple[int, str]]:
    """
    例: "1\tこんにちは" / "1: こんにちは"
    """
    if not line:
        return None
    line = line.strip("\r")
    if not line.strip():
        return None
    m = _INDEXED_LINE_RE.match(line)
    if not m:
        return None
    idx = int(m.group(1))
    ja = m.group(2).strip()
    if not ja:
        return None
    return idx, ja


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    LLM の出力から最初の JSON オブジェクト部分を抜き出す（前後に余計な文字が付くケース対策）。
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # .env から region を渡す想定（未設定ならデフォルト）
    region = os.getenv("AWS_REGION", AWS_REGION_DEFAULT)

    client = get_bedrock_client(region)
    # キューを浅くしてリアルタイム性を確保（20フレーム = 2秒分、古いフレームは破棄）
    audio_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=20)
    translation_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=50)
    catchup_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)

    # シンプルなプロンプトで低遅延化
    system_prompt = "Transcribe the user's speech into text accurately."

    session_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    session: Optional[NovaSonicSession] = None
    output_task: Optional[asyncio.Task] = None

    async def safe_send(payload: dict) -> bool:
        """
        WebSocket 切断後に background task が send してしまうケースを安全に無視する。
        """
        if stop_event.is_set():
            return False
        try:
            await websocket.send_text(json.dumps(payload))
            return True
        except RuntimeError as e:
            # 例: Unexpected ASGI message 'websocket.send', after sending 'websocket.close'
            if "Unexpected ASGI message" in str(e):
                return False
            return False
        except Exception:
            return False

    # 会議ログ（USER final を時刻付きで保持。翻訳が届いたら ja を追記）
    utterance_log: deque[dict] = deque()
    utterance_by_id: dict[str, dict] = {}

    # 統計情報
    audio_frame_count = 0
    last_audio_time = time.time()
    transcription_count = 0
    translation_count = 0
    translation_started = False
    translation_enabled = True
    translation_disabled_reason: Optional[str] = None
    translation_segment_seq = 0
    translation_model_id_in_use = TRANSLATION_MODEL_ID
    translation_model_id_fallback = TRANSLATION_MODEL_ID_FALLBACK
    conversation_epoch = 0

    catchup_count = 0
    catchup_model_id_in_use = CATCHUP_MODEL_ID
    catchup_model_id_fallback = CATCHUP_MODEL_ID_FALLBACK

    logger.info(
        "Translation model configured: "
        f"primary='{translation_model_id_in_use}', fallback='{translation_model_id_fallback}'"
    )
    logger.info(
        "Catch-up model configured: "
        f"primary='{catchup_model_id_in_use}', fallback='{catchup_model_id_fallback}'"
    )

    def _prune_utterance_log(now_ts: float) -> None:
        while len(utterance_log) > CATCHUP_LOG_MAX_ITEMS:
            old = utterance_log.popleft()
            try:
                utterance_by_id.pop(old.get("id"), None)
            except Exception:
                pass
        while utterance_log:
            oldest = utterance_log[0]
            if now_ts - float(oldest.get("ts", now_ts)) <= CATCHUP_LOG_MAX_SECONDS:
                break
            old = utterance_log.popleft()
            try:
                utterance_by_id.pop(old.get("id"), None)
            except Exception:
                pass

    async def translation_worker():
        nonlocal translation_count, translation_model_id_in_use, translation_enabled, translation_disabled_reason, conversation_epoch
        logger.info("translation_worker task started, waiting for translation requests...")
        last_request_at = 0.0
        backoff_seconds = 0.0
        while not stop_event.is_set():
            try:
                first_item = await translation_q.get()
            except asyncio.CancelledError:
                break

            if stop_event.is_set():
                break
            if not translation_enabled:
                continue
            if first_item.get("epoch") != conversation_epoch:
                continue

            # Debounce: 少し待って複数の final セグメントをまとめて翻訳（リクエスト数削減）
            batch_items = [first_item]
            batch_chars = len((first_item.get("en") or "").strip())
            batch_epoch = first_item.get("epoch")
            start = time.monotonic()
            while batch_chars < TRANSLATION_MAX_BATCH_CHARS:
                remaining = TRANSLATION_DEBOUNCE_SECONDS - (time.monotonic() - start)
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(translation_q.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return
                if not nxt:
                    continue
                if nxt.get("epoch") != batch_epoch:
                    continue
                batch_items.append(nxt)
                batch_chars += len((nxt.get("en") or "").strip())

            pending_items = []
            for it in batch_items:
                seg_id = it.get("id")
                en = (it.get("en") or "").replace("\n", " ").strip()
                if not seg_id or not en:
                    continue
                pending_items.append({"id": seg_id, "en": en})
            if not pending_items:
                continue
            if batch_epoch != conversation_epoch:
                continue

            translation_count += 1
            logger.info(
                f"Translation request #{translation_count}: "
                f"segments={len(pending_items)}, text_len={sum(len(x['en']) for x in pending_items)}"
            )

            while pending_items and not stop_event.is_set():
                # リクエスト間隔を空ける（Throttling 対策）
                now = time.monotonic()
                wait_for = (last_request_at + TRANSLATION_MIN_INTERVAL_SECONDS) - now
                if wait_for > 0:
                    await asyncio.sleep(wait_for)

                last_request_at = time.monotonic()

                try:
                    if not await safe_send({"type": "status", "status": "translation_translating"}):
                        return

                    # バッチ入力を "N<TAB>English" で渡し、出力も同形式を要求する（英日対応づけ用）
                    input_lines = []
                    for i, it in enumerate(pending_items, start=1):
                        input_lines.append(f"{i}\t{it['en']}")

                    prompt = (
                        "Translate each line from English to natural Japanese.\n"
                        "Each input line is formatted as: N<TAB>English.\n"
                        "Output exactly one line per input line, formatted as: N<TAB>Japanese.\n"
                        "Keep N the same. Do not add any extra lines, headings, or code blocks.\n\n"
                        + "\n".join(input_lines)
                    )

                    buffer = ""
                    async for delta in stream_prompt_text_deltas(
                        client, prompt=prompt, model_id=translation_model_id_in_use
                    ):
                        if batch_epoch != conversation_epoch:
                            buffer = ""
                            pending_items = []
                            break
                        buffer += delta
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            parsed = _parse_indexed_output_line(line)
                            if not parsed:
                                continue
                            idx, ja = parsed
                            if 1 <= idx <= len(pending_items):
                                seg_id = pending_items[idx - 1]["id"]
                                try:
                                    ut = utterance_by_id.get(seg_id)
                                    if ut is not None:
                                        ut["ja"] = ja
                                except Exception:
                                    pass
                                if not await safe_send({"type": "aligned_ja", "id": seg_id, "ja": ja}):
                                    return

                    # 最後に改行が来ないケース
                    parsed_last = _parse_indexed_output_line(buffer)
                    if parsed_last:
                        idx, ja = parsed_last
                        if 1 <= idx <= len(pending_items):
                            seg_id = pending_items[idx - 1]["id"]
                            try:
                                ut = utterance_by_id.get(seg_id)
                                if ut is not None:
                                    ut["ja"] = ja
                            except Exception:
                                pass
                            if not await safe_send({"type": "aligned_ja", "id": seg_id, "ja": ja}):
                                return

                    backoff_seconds = 0.0
                    if not await safe_send({"type": "status", "status": "translation_idle"}):
                        return
                    break

                except ResourceNotFoundException as e:
                    message_raw = getattr(e, "message", None) or str(e) or ""
                    message = message_raw.lower()
                    if "use case details" in message and "anthropic" in message:
                        await safe_send({"type": "status", "status": "translation_disabled"})
                        translation_enabled = False
                        translation_disabled_reason = message_raw
                        logger.error(
                            "Translation disabled (Anthropic use case details not submitted). "
                            f"details='{message_raw}'"
                        )
                        try:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "translation_error",
                                        "error": (
                                            "Anthropic の利用目的（use case details）が未提出のため翻訳を実行できません。"
                                            "AWS コンソール → Bedrock → Model access で Anthropic の use case details を提出し、"
                                            "15分ほど待ってから再試行してください。"
                                        ),
                                    }
                                )
                            )
                        except Exception:
                            break

                        # 以降の翻訳リクエストを破棄（ログ/エラーのスパム防止）
                        try:
                            while True:
                                translation_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        pending_items = []
                        break

                    logger.error(f"Translation error: {e}", exc_info=True)
                    await safe_send({"type": "status", "status": "translation_error"})
                    await safe_send({"type": "translation_error", "error": str(e)})
                    break

                except ValidationException as e:
                    message = (getattr(e, "message", None) or str(e) or "").lower()
                    if (
                        "on-demand throughput" in message
                        and "inference profile" in message
                        and translation_model_id_in_use != translation_model_id_fallback
                    ):
                        logger.warning(
                            "Translation model does not support on-demand throughput. "
                            f"Switching to inference profile: {translation_model_id_fallback}"
                        )
                        translation_model_id_in_use = translation_model_id_fallback
                        try:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "info",
                                        "text": (
                                            "翻訳モデルを inference profile に切り替えました: "
                                            f"model_id={translation_model_id_in_use}"
                                        ),
                                    }
                                )
                            )
                        except Exception:
                            pass
                        continue

                    logger.error(f"Translation error: {e}", exc_info=True)
                    await safe_send({"type": "status", "status": "translation_error"})
                    await safe_send({"type": "translation_error", "error": str(e)})
                    break

                except ThrottlingException as e:
                    # 翻訳が追いつかないときは待って再試行し、キューが溜まっていればまとめて翻訳する
                    backoff_seconds = min(backoff_seconds * 2 if backoff_seconds else 0.5, 8.0)
                    logger.warning(
                        f"Translation throttled. Backing off {backoff_seconds:.1f}s: {e}"
                    )
                    await safe_send({"type": "status", "status": "translation_throttled"})
                    await safe_send(
                        {
                            "type": "info",
                            "text": f"翻訳が混雑しています。{backoff_seconds:.1f}秒待って再試行します…",
                        }
                    )

                    # 待機中に溜まった分をまとめる（ただしサイズ上限まで）
                    try:
                        while sum(len(x["en"]) for x in pending_items) < TRANSLATION_MAX_BATCH_CHARS:
                            nxt = translation_q.get_nowait()
                            if not nxt:
                                continue
                            if nxt.get("epoch") != batch_epoch:
                                continue
                            seg_id = nxt.get("id")
                            en = (nxt.get("en") or "").replace("\n", " ").strip()
                            if not seg_id or not en:
                                continue
                            if sum(len(x["en"]) for x in pending_items) + len(en) > TRANSLATION_MAX_BATCH_CHARS:
                                break
                            pending_items.append({"id": seg_id, "en": en})
                    except asyncio.QueueEmpty:
                        pass

                    await asyncio.sleep(backoff_seconds)
                    continue

                except Exception as e:
                    logger.error(f"Translation error: {e}", exc_info=True)
                    await safe_send({"type": "status", "status": "translation_error"})
                    await safe_send({"type": "translation_error", "error": str(e)})
                    break

        logger.info(
            "translation_worker task ended. "
            f"Total translations: {translation_count}, stop_event: {stop_event.is_set()}"
        )

    async def catchup_worker():
        nonlocal catchup_count, catchup_model_id_in_use, conversation_epoch
        logger.info("catchup_worker task started, waiting for catch-up requests...")
        last_request_at = 0.0
        while not stop_event.is_set():
            try:
                req = await catchup_q.get()
            except asyncio.CancelledError:
                break

            if stop_event.is_set():
                break
            if req.get("epoch") != conversation_epoch:
                continue

            seconds = int(req.get("seconds") or 120)
            seconds = max(10, min(seconds, CATCHUP_LOG_MAX_SECONDS))
            batch_epoch = req.get("epoch")

            # リクエスト間隔を空ける（翻訳とは別に抑制）
            now_mono = time.monotonic()
            wait_for = (last_request_at + CATCHUP_MIN_INTERVAL_SECONDS) - now_mono
            if wait_for > 0:
                await safe_send({"type": "status", "status": "catchup_generating"})
                await safe_send(
                    {
                        "type": "info",
                        "text": f"Catch up は短時間に連続実行できません（{wait_for:.0f}秒後に実行します）…",
                    }
                )
                await asyncio.sleep(wait_for)

            last_request_at = time.monotonic()
            if batch_epoch != conversation_epoch:
                continue

            if not await safe_send({"type": "status", "status": "catchup_generating"}):
                return

            now_ts = time.time()
            cutoff = now_ts - seconds
            _prune_utterance_log(now_ts)

            window_items = [u for u in utterance_log if float(u.get("ts", 0)) >= cutoff]
            if not window_items:
                if not await safe_send(
                    {
                        "type": "catchup_result",
                        "window_seconds": seconds,
                        "topic": "",
                        "important_points": [],
                        "decisions": [],
                        "next_topic": "",
                    }
                ):
                    return
                await safe_send({"type": "status", "status": "catchup_ready"})
                continue

            # 入力が長すぎる場合は末尾（最新）から詰める
            chosen: list[dict] = []
            total_chars = 0
            for u in reversed(window_items):
                en = (u.get("en") or "").strip()
                ja = (u.get("ja") or "").strip()
                block = f"{u.get('id')}\nEN: {en}\n"
                if ja:
                    block += f"JA: {ja}\n"
                block += "\n"
                if total_chars + len(block) > CATCHUP_MAX_INPUT_CHARS and chosen:
                    break
                chosen.append(u)
                total_chars += len(block)
            chosen.reverse()

            transcript_blocks = []
            for u in chosen:
                seg_id = u.get("id")
                en = (u.get("en") or "").strip()
                ja = (u.get("ja") or "").strip()
                if not seg_id or not en:
                    continue
                transcript_blocks.append(f"{seg_id}\nEN: {en}")
                if ja:
                    transcript_blocks.append(f"JA: {ja}")
                transcript_blocks.append("")
            transcript_text = "\n".join(transcript_blocks).strip()

            catchup_count += 1
            logger.info(
                f"Catch up request #{catchup_count}: window={seconds}s, segments={len(chosen)}, chars={len(transcript_text)}"
            )

            prompt = (
                "You are a meeting catch-up assistant.\n"
                "A user lost track of the conversation and needs a quick catch-up.\n\n"
                "You will receive transcript segments with IDs like u12.\n"
                "Each segment may have EN and sometimes JA.\n"
                "Use JA when available; otherwise use EN.\n\n"
                "Return ONLY valid JSON (no markdown), with this schema:\n"
                "{\n"
                '  "topic": string,\n'
                '  "important_points": [{"text": string, "ids": ["u12", "..."]}],\n'
                '  "decisions": [{"text": string, "ids": ["u12", "..."]}],\n'
                '  "next_topic": string\n'
                "}\n\n"
                "Rules:\n"
                "- All strings must be Japanese.\n"
                "- Keep it concise and practical for a meeting.\n"
                "- For each bullet, include 1-3 supporting ids when possible.\n"
                "- Do not invent facts not present in the transcript.\n\n"
                f"Time window: last {seconds} seconds.\n\n"
                "Transcript:\n"
                f"{transcript_text}\n"
            )

            try:
                buf = ""
                async for delta in stream_prompt_text_deltas(
                    client, prompt=prompt, model_id=catchup_model_id_in_use
                ):
                    if batch_epoch != conversation_epoch:
                        buf = ""
                        break
                    buf += delta

                if batch_epoch != conversation_epoch or not buf.strip():
                    continue

                json_part = _extract_first_json_object(buf)
                parsed = json.loads(json_part) if json_part else None
                if not isinstance(parsed, dict):
                    raise ValueError("Invalid JSON from model")

                topic = str(parsed.get("topic") or "")
                important_points = parsed.get("important_points") or []
                decisions = parsed.get("decisions") or []
                next_topic = str(parsed.get("next_topic") or "")

                def _norm_items(items):
                    out = []
                    if not isinstance(items, list):
                        return out
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        text = str(it.get("text") or "").strip()
                        ids = it.get("ids") or []
                        if not isinstance(ids, list):
                            ids = []
                        ids = [str(x) for x in ids if isinstance(x, (str, int))]
                        ids = [i for i in ids if i.startswith("u")]
                        if text:
                            out.append({"text": text, "ids": ids[:3]})
                    return out

                msg = {
                    "type": "catchup_result",
                    "window_seconds": seconds,
                    "topic": topic,
                    "important_points": _norm_items(important_points),
                    "decisions": _norm_items(decisions),
                    "next_topic": next_topic,
                }
                if not await safe_send(msg):
                    return
                await safe_send({"type": "status", "status": "catchup_ready"})

            except ValidationException as e:
                message = (getattr(e, "message", None) or str(e) or "").lower()
                if (
                    "on-demand throughput" in message
                    and "inference profile" in message
                    and catchup_model_id_in_use != catchup_model_id_fallback
                ):
                    logger.warning(
                        "Catch-up model does not support on-demand throughput. "
                        f"Switching to inference profile: {catchup_model_id_fallback}"
                    )
                    catchup_model_id_in_use = catchup_model_id_fallback
                    try:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "info",
                                    "text": (
                                        "Catch up モデルを inference profile に切り替えました: "
                                        f"model_id={catchup_model_id_in_use}"
                                    ),
                                }
                            )
                        )
                    except Exception:
                        pass
                    continue

                logger.error(f"Catch up error: {e}", exc_info=True)
                await safe_send({"type": "status", "status": "catchup_error"})
                await safe_send({"type": "catchup_error", "error": str(e)})

            except ThrottlingException as e:
                logger.warning(f"Catch up throttled: {e}")
                await safe_send({"type": "status", "status": "catchup_throttled"})
                await safe_send(
                    {
                        "type": "catchup_error",
                        "error": "混雑しています。少し待ってから再試行してください。",
                    }
                )

            except Exception as e:
                logger.error(f"Catch up error: {e}", exc_info=True)
                await safe_send({"type": "status", "status": "catchup_error"})
                await safe_send({"type": "catchup_error", "error": str(e)})

        logger.info("catchup_worker task ended.")

    async def start_session():
        nonlocal session, output_task
        try:
            s = NovaSonicSession(client, MODEL_ID)
            await s.open(system_prompt=system_prompt)
            session = s
            status_msg = json.dumps({"type": "status", "status": "aws_connected"})
            await websocket.send_text(status_msg)
            logger.debug(f"Sent to browser: {status_msg}")
        except Exception as e:
            logger.error(f"Failed to start Bedrock session: {e}")
            error_msg = json.dumps({"type": "status", "status": "aws_error", "error": str(e)})
            await websocket.send_text(error_msg)
            logger.debug(f"Sent to browser: {error_msg}")
            raise

        if output_task and not output_task.done():
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)

        output_task = asyncio.create_task(read_outputs())

    async def read_outputs():
        nonlocal session, transcription_count, translation_segment_seq
        logger.info("read_outputs task started, waiting for Bedrock events...")
        event_count = 0
        while not stop_event.is_set() and session and session.is_active:
            try:
                msg = await session.recv_once()
                if not msg:
                    logger.debug("Received None from Bedrock")
                    continue

                event_count += 1
                logger.debug(f"Received message #{event_count} from Bedrock")

                if "event" not in msg:
                    logger.warning(f"Received message without 'event' key: {msg}")
                    continue

                ev = msg["event"]
                logger.debug(f"Bedrock event #{event_count}: {list(ev.keys())}")

                if "contentStart" in ev:
                    cs = ev["contentStart"]
                    session.current_role = cs.get("role")
                    session.current_generation_stage = _extract_generation_stage(cs) or "FINAL"
                    logger.info(f"Content started: role={session.current_role}, stage={session.current_generation_stage}")
                    continue

                if "textOutput" in ev:
                    text = ev["textOutput"].get("content", "")
                    role = session.current_role
                    stage = session.current_generation_stage
                    logger.info(f"Received textOutput: role={role}, stage={stage}, text_len={len(text)}, text='{text[:100]}'")

                    if not text:
                        logger.warning("Empty text in textOutput")
                        continue

                    # Nova 側が返す可能性があるシステムタグ（例: {"interrupted": true}）は表示/翻訳対象外
                    if _is_interrupted_tag(text):
                        logger.info(f"Skipping system tag textOutput: {text.strip()}")
                        continue

                    # USERロールとASSISTANTロールの両方を表示（Nova 2 Sonicは会話型AIのため）
                    if session.current_role in ["USER", "ASSISTANT"]:
                        stage_lower = (session.current_generation_stage or "FINAL").lower()
                        transcription_count += 1
                        logger.info(f"Transcription #{transcription_count} ({stage_lower}, {role}): {text[:50]}...")
                        await websocket.send_text(json.dumps({"type": stage_lower, "text": text}))
                        await websocket.send_text(json.dumps({"type": "status", "status": "transcribing"}))

                        # final（確定）テキストのみ翻訳（レイテンシ/コスト最適化）
                        # NOTE: 翻訳は「音声入力＝USERロール」のみ（ASSISTANTは翻訳しない）
                        if stage_lower == "final" and role == "USER":
                            if not translation_enabled:
                                continue
                            try:
                                translation_segment_seq += 1
                                seg_id = f"u{translation_segment_seq}"
                                now_ts = time.time()
                                ut = {
                                    "id": seg_id,
                                    "ts": now_ts,
                                    "en": text,
                                    "ja": "",
                                    "epoch": conversation_epoch,
                                }
                                utterance_log.append(ut)
                                utterance_by_id[seg_id] = ut
                                _prune_utterance_log(now_ts)
                                await websocket.send_text(
                                    json.dumps({"type": "aligned_en", "id": seg_id, "en": text})
                                )
                                translation_q.put_nowait({"epoch": conversation_epoch, "id": seg_id, "en": text})
                            except asyncio.QueueFull:
                                # キューが満杯の場合、古いリクエストを破棄して新しい翻訳を優先（リアルタイム性優先）
                                try:
                                    translation_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                try:
                                    translation_segment_seq += 1
                                    seg_id = f"u{translation_segment_seq}"
                                    now_ts = time.time()
                                    ut = {
                                        "id": seg_id,
                                        "ts": now_ts,
                                        "en": text,
                                        "ja": "",
                                        "epoch": conversation_epoch,
                                    }
                                    utterance_log.append(ut)
                                    utterance_by_id[seg_id] = ut
                                    _prune_utterance_log(now_ts)
                                    await websocket.send_text(
                                        json.dumps({"type": "aligned_en", "id": seg_id, "en": text})
                                    )
                                    translation_q.put_nowait({"epoch": conversation_epoch, "id": seg_id, "en": text})
                                    logger.warning(
                                        "Translation queue full, dropped oldest translation request to insert new one"
                                    )
                                except asyncio.QueueFull:
                                    logger.warning("Translation queue full, dropped translation request")
                    else:
                        logger.debug(f"Skipping non-USER/ASSISTANT text (role={role})")
                    continue

                # その他のイベントもログに記録
                if "error" in ev:
                    logger.error(f"Bedrock error event: {ev['error']}")
                elif "sessionEnd" in ev:
                    logger.info("Bedrock session ended event received")
                elif "audioOutput" in ev:
                    logger.debug("Audio output received (ignored)")
                else:
                    logger.debug(f"Other Bedrock event: {ev}")

            except Exception as e:
                logger.error(f"Error reading AWS output: {e}", exc_info=True)
                await websocket.send_text(json.dumps({"type": "status", "status": "aws_error", "error": str(e)}))
                break

        logger.info(f"read_outputs task ended. Total events received: {event_count}, stop_event: {stop_event.is_set()}, session active: {session.is_active if session else 'None'}")

    async def renew_loop():
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=RENEW_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                return
            logger.info("Renewing Bedrock session (8-minute limit prevention)...")
            async with session_lock:
                if session:
                    await session.close()
                await start_session()
            await websocket.send_text(json.dumps({"type": "info", "text": "Session renewed to avoid the 8-minute limit."}))

    async def keepalive_loop():
        """プロキシタイムアウト回避用のキープアライブ"""
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=KEEPALIVE_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                return
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
                logger.debug(f"Sent keepalive ping (audio frames: {audio_frame_count})")
            except Exception as e:
                logger.warning(f"Keepalive ping failed: {e}")
                break

    async def recv_from_browser():
        nonlocal audio_frame_count, last_audio_time, translation_started, translation_segment_seq, conversation_epoch
        try:
            while not stop_event.is_set():
                message = await websocket.receive()

                # テキストメッセージ（JSON）の処理
                if message["type"] == "websocket.receive" and "text" in message:
                    try:
                        msg = json.loads(message["text"])
                        if msg.get("type") == "pong":
                            logger.debug("Received pong from browser")
                        elif msg.get("type") == "clear":
                            # ブラウザ側で表示がクリアされたので、区切り用の先頭改行を抑制し、
                            # 追いかけている翻訳キューも捨てて表示と整合させる
                            translation_started = False
                            conversation_epoch += 1
                            translation_segment_seq = 0
                            utterance_log.clear()
                            utterance_by_id.clear()
                            try:
                                await websocket.send_text(json.dumps({"type": "status", "status": "translation_idle"}))
                                await websocket.send_text(json.dumps({"type": "status", "status": "catchup_idle"}))
                            except Exception:
                                break
                            try:
                                while True:
                                    translation_q.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            try:
                                while True:
                                    catchup_q.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        elif msg.get("type") == "catchup":
                            seconds = int(msg.get("seconds") or 120)
                            seconds = max(10, min(seconds, CATCHUP_LOG_MAX_SECONDS))
                            req = {"epoch": conversation_epoch, "seconds": seconds}
                            try:
                                catchup_q.put_nowait(req)
                            except asyncio.QueueFull:
                                try:
                                    catchup_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                try:
                                    catchup_q.put_nowait(req)
                                except asyncio.QueueFull:
                                    pass
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {message['text'][:100]}")

                # バイナリメッセージ（音声データ）の処理
                elif message["type"] == "websocket.receive" and "bytes" in message:
                    frame = message["bytes"]
                    try:
                        audio_q.put_nowait(frame)
                        audio_frame_count += 1
                        current_time = time.time()
                        if current_time - last_audio_time >= 5.0:
                            logger.info(f"Audio receiving: {audio_frame_count} frames received")
                            await websocket.send_text(json.dumps({"type": "status", "status": "audio_receiving"}))
                            last_audio_time = current_time
                    except asyncio.QueueFull:
                        # キューが満杯の場合、古いフレームを破棄して新しいフレームを追加（リアルタイム性優先）
                        try:
                            audio_q.get_nowait()  # 古いフレームを破棄
                            audio_q.put_nowait(frame)  # 新しいフレームを追加
                            logger.warning("Audio queue full, dropped oldest frame to insert new frame")
                        except Exception as e:
                            logger.warning(f"Failed to drop old frame: {e}")

                # 切断メッセージ
                elif message["type"] == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error in recv_from_browser: {e}")
        finally:
            stop_event.set()
            try:
                audio_q.put_nowait(None)
            except Exception:
                pass
            logger.info(f"Browser disconnected. Total audio frames: {audio_frame_count}, transcriptions: {transcription_count}")

    async def send_to_bedrock():
        nonlocal audio_frame_count
        frame_count_local = 0
        total_bytes = 0
        while not stop_event.is_set():
            frame = await audio_q.get()
            if frame is None:
                logger.info(f"Audio stream ended. Total frames sent: {frame_count_local}, total bytes: {total_bytes}")
                return
            # ロックフリーでセッション参照を取得（レイテンシ削減）
            # セッション差し替え時のみ session_lock を使用（renew_loop 内）
            current_session = session
            if current_session and current_session.is_active:
                try:
                    await current_session.send_audio(frame)
                    frame_count_local += 1
                    total_bytes += len(frame)
                    if frame_count_local == 1:
                        logger.info(f"First audio frame sent to Bedrock: {len(frame)} bytes")
                    elif frame_count_local % 100 == 0:
                        logger.info(f"Sent {frame_count_local} audio frames to Bedrock ({total_bytes} bytes)")
                except Exception as e:
                    logger.error(f"Error sending audio to Bedrock: {e}", exc_info=True)
                    await websocket.send_text(json.dumps({"type": "status", "status": "aws_error", "error": str(e)}))

    async with session_lock:
        await start_session()

    tasks = [
        asyncio.create_task(renew_loop()),
        asyncio.create_task(keepalive_loop()),
        asyncio.create_task(recv_from_browser()),
        asyncio.create_task(send_to_bedrock()),
        asyncio.create_task(translation_worker()),
        asyncio.create_task(catchup_worker()),
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        for t in tasks + ([output_task] if output_task else []):
            if t and not t.done():
                t.cancel()
        await asyncio.gather(*(t for t in tasks if t), return_exceptions=True)
        if output_task:
            await asyncio.gather(output_task, return_exceptions=True)

        async with session_lock:
            if session:
                await session.close()
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Nova 2 Sonic – Real-time Transcription (English)</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0f17;
      --bg-elev: #0f172a;
      --panel: #0f172a;
      --panel-2: #0b1220;
      --border: #243244;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --muted-2: #6b7280;
      --code-bg: #111827;
      --btn-bg: #111827;
      --btn-hover: #1f2937;
      --btn-border: #2b3648;
      --shadow: 0 1px 0 rgba(0,0,0,0.25);
    }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 20px; background: var(--bg); color: var(--text); }
    h1 { margin: 0 0 6px 0; font-size: 20px; color: var(--text); }
    .sub { color: var(--muted); margin-bottom: 14px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; align-items: center; }
    button {
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
      color: var(--text);
      background: var(--btn-bg);
      border: 1px solid var(--btn-border);
      border-radius: 10px;
      box-shadow: var(--shadow);
    }
    button:hover { background: var(--btn-hover); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    #status { font-size: 13px; color: var(--muted); }
    select {
      padding: 10px 12px;
      font-size: 14px;
      color: var(--text);
      background: var(--btn-bg);
      border: 1px solid var(--btn-border);
      border-radius: 10px;
      box-shadow: var(--shadow);
    }
    select:disabled { opacity: 0.55; }
    option { background: var(--btn-bg); color: var(--text); }
    .sticky-bar {
      position: sticky;
      top: 0;
      z-index: 50;
      background: var(--bg);
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
    }
    .sticky-bar .controls { margin: 0; }
    .sticky-bar .status-indicators { margin: 10px 0 0 0; }
    .status-indicators { display: flex; gap: 12px; margin: 10px 0; font-size: 13px; }
    .indicator { display: flex; align-items: center; gap: 6px; color: var(--muted); }
    .indicator-dot { width: 10px; height: 10px; border-radius: 50%; background: #4b5563; }
    .indicator-dot.active { background: #22c55e; animation: pulse 2s infinite; }
    .indicator-dot.error { background: #ef4444; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    #transcript {
      white-space: pre-wrap;
      min-height: 360px;
      max-height: 500px;
      overflow-y: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 10px;
      font-size: 16px;
      line-height: 1.5;
    }
    #translation {
      white-space: pre-wrap;
      min-height: 360px;
      max-height: 500px;
      overflow-y: auto;
      background: var(--panel-2);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 10px;
      font-size: 16px;
      line-height: 1.5;
      margin-top: 12px;
    }
    #pairs {
      min-height: 360px;
      max-height: 500px;
      overflow-y: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-top: 12px;
    }
    .pair {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .pair:hover { background: rgba(255,255,255,0.04); }
    .pair:last-child { border-bottom: 0; }
    .pair-col { white-space: pre-wrap; }
    .pair-col::before { display: none; }
    .pair-en { color: var(--text); }
    .pair-ja { color: #5eead4; }
    @media (max-width: 720px) {
      .pair { grid-template-columns: 1fr; gap: 10px; }
      .pair-col::before {
        display: block;
        content: attr(data-label);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: var(--muted-2);
        margin-bottom: 6px;
      }
    }
    #catchup {
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 10px;
      margin-top: 12px;
      font-size: 14px;
      line-height: 1.5;
    }
    #catchup ul { margin: 8px 0 0 0; padding-left: 18px; }
    .catchup-meta { color: var(--muted); font-size: 12px; }
    .catchup-refs { margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }
    .ref-chip {
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(99,102,241,0.14);
      color: #c7d2fe;
      cursor: pointer;
      user-select: none;
    }
    .ref-chip:hover { background: rgba(99,102,241,0.22); }
    .pair.selected { background: rgba(245,158,11,0.14); box-shadow: inset 0 0 0 1px rgba(245,158,11,0.45); }
    .hint { margin-top: 10px; font-size: 12px; color: var(--muted); }
    code { background: var(--code-bg); padding: 1px 4px; border-radius: 4px; color: var(--text); border: 1px solid rgba(255,255,255,0.08); }
  </style>
</head>
<body>
  <h1>Amazon Nova 2 Sonic – Real-time Transcription</h1>
  <div class="sub">Language: <b>English</b> (audio input: 16 kHz, mono, 16-bit PCM)</div>

  <div class="sticky-bar">
    <div class="controls">
      <button id="start">Start</button>
      <button id="stop" disabled>Stop</button>
      <button id="download" disabled>Download TXT</button>
      <button id="clear" disabled>Clear</button>
      <span style="margin-left: 8px; font-size: 12px; color: var(--muted);">Input:</span>
      <select id="input-source">
        <option value="mic" selected>Mic</option>
        <option value="tab">Tab Audio (Chrome/Edge)</option>
      </select>
      <span style="margin-left: 8px; font-size: 12px; color: var(--muted);">Catch up:</span>
      <button id="catchup30" disabled>30s</button>
      <button id="catchup120" disabled>2m</button>
      <button id="catchup300" disabled>5m</button>
      <div id="status"></div>
    </div>

    <div class="status-indicators">
      <div class="indicator">
        <div class="indicator-dot" id="ws-indicator"></div>
        <span>WebSocket: <span id="ws-status">Disconnected</span></span>
      </div>
      <div class="indicator">
        <div class="indicator-dot" id="audio-indicator"></div>
        <span>Audio: <span id="audio-status">Idle</span></span>
      </div>
      <div class="indicator">
        <div class="indicator-dot" id="aws-indicator"></div>
        <span>AWS: <span id="aws-status">Idle</span></span>
      </div>
      <div class="indicator">
        <div class="indicator-dot" id="translation-indicator"></div>
        <span>Translation: <span id="translation-status">Idle</span></span>
      </div>
      <div class="indicator">
        <div class="indicator-dot" id="catchup-indicator"></div>
        <span>Catch up: <span id="catchup-status">Idle</span></span>
      </div>
    </div>
  </div>

  <div id="transcript">Press "Start" and speak.</div>
  <h3>Japanese Translation</h3>
  <div id="translation">…</div>
  <h3>Aligned EN ↔ JA (USER final)</h3>
  <div id="pairs">…</div>
  <h3>Catch up</h3>
  <div id="catchup">Press a button (30s / 2m / 5m) to catch up.</div>
  <div class="hint">
    If you deploy behind HTTPS, this page will automatically use <code>wss://</code>. Microphone access requires a secure context (HTTPS or localhost).
    <br>Status indicators show connection health in real-time.
  </div>

<script>
	(() => {
	  let ws = null;
	  let audioContext = null;
      let captureStream = null;
	  let sourceNode = null;
	  let workletNode = null;
	  let legacyProcessor = null;
	  let pingInterval = null;
	  let wakeLock = null;
	  let useAudioWorklet = false;
      let stopping = false;

  let finalText = "";
  let partialText = "";
	  let finalTranslation = "";
	  let partialTranslation = "";
      const alignedPairs = new Map(); // id -> { en, ja, row, enEl, jaEl }

  const TARGET_SR = 16000;
  const FRAME_SAMPLES = 1600; // 100ms at 16kHz

  // リングバッファ：固定サイズで再割り当てを防止
  const RING_BUFFER_SIZE = 8000; // 500ms分のバッファ
  const ringBuffer = new Int16Array(RING_BUFFER_SIZE);
  let ringWriteIndex = 0;
  let ringReadIndex = 0;

  // フレーム送信用の固定バッファ（再割り当て防止）
  const frameBuffer = new ArrayBuffer(FRAME_SAMPLES * 2);
  const frameView = new DataView(frameBuffer);

	  const transcriptDiv = document.getElementById("transcript");
	  const translationDiv = document.getElementById("translation");
      const pairsDiv = document.getElementById("pairs");
	  const statusDiv = document.getElementById("status");
	  const startBtn = document.getElementById("start");
		  const stopBtn = document.getElementById("stop");
		  const downloadBtn = document.getElementById("download");
		  const clearBtn = document.getElementById("clear");
      const catchup30Btn = document.getElementById("catchup30");
      const catchup120Btn = document.getElementById("catchup120");
      const catchup300Btn = document.getElementById("catchup300");
      const catchupDiv = document.getElementById("catchup");
      const inputSourceSel = document.getElementById("input-source");

  const wsIndicator = document.getElementById("ws-indicator");
  const wsStatus = document.getElementById("ws-status");
  const audioIndicator = document.getElementById("audio-indicator");
	  const audioStatus = document.getElementById("audio-status");
	  const awsIndicator = document.getElementById("aws-indicator");
	  const awsStatus = document.getElementById("aws-status");
      const translationIndicator = document.getElementById("translation-indicator");
      const translationStatus = document.getElementById("translation-status");
      const catchupIndicator = document.getElementById("catchup-indicator");
      const catchupStatus = document.getElementById("catchup-status");

  function setStatus(msg) { statusDiv.textContent = msg || ""; }

	  function updateIndicator(type, status, text) {
	    let indicator, statusSpan;
	    if (type === "ws") { indicator = wsIndicator; statusSpan = wsStatus; }
	    else if (type === "audio") { indicator = audioIndicator; statusSpan = audioStatus; }
	    else if (type === "aws") { indicator = awsIndicator; statusSpan = awsStatus; }
        else if (type === "translation") { indicator = translationIndicator; statusSpan = translationStatus; }
        else if (type === "catchup") { indicator = catchupIndicator; statusSpan = catchupStatus; }
        else { return; }

	    indicator.className = "indicator-dot " + status;
	    statusSpan.textContent = text;
	  }

  function updateView() {
    const merged = finalText + (partialText ? partialText : "");
    transcriptDiv.textContent = merged || "…";
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
  }

	  function updateTranslationView() {
	    translationDiv.textContent = finalTranslation || "…";
	    translationDiv.scrollTop = translationDiv.scrollHeight;
	  }

      function rebuildTranslationFromPairs() {
        const lines = [];
        for (const item of alignedPairs.values()) {
          if (item.ja) lines.push(item.ja);
        }
        finalTranslation = lines.join("\\n");
        partialTranslation = "";
        updateTranslationView();
      }

      function addAlignedPair(id, en) {
        if (!id) return;
        if (alignedPairs.has(id)) return;
        if (pairsDiv.textContent === "…") pairsDiv.textContent = "";

        const row = document.createElement("div");
        row.className = "pair";
        row.dataset.id = id;

        const enEl = document.createElement("div");
        enEl.className = "pair-col pair-en";
        enEl.dataset.label = "EN";
        enEl.textContent = en || "";

        const jaEl = document.createElement("div");
        jaEl.className = "pair-col pair-ja";
        jaEl.dataset.label = "JA";
        jaEl.textContent = "…";

        row.appendChild(enEl);
        row.appendChild(jaEl);
        pairsDiv.appendChild(row);

        alignedPairs.set(id, { en: en || "", ja: "", row, enEl, jaEl });
        pairsDiv.scrollTop = pairsDiv.scrollHeight;
      }

      function setAlignedJa(id, ja) {
        if (!id) return;
        if (!alignedPairs.has(id)) addAlignedPair(id, "");
        const item = alignedPairs.get(id);
        item.ja = ja || "";
        item.jaEl.textContent = item.ja || "…";
        pairsDiv.scrollTop = pairsDiv.scrollHeight;
        rebuildTranslationFromPairs();
      }

      let selectedPairId = null;
      function highlightPair(id) {
        if (!id) return;
        if (selectedPairId && alignedPairs.has(selectedPairId)) {
          alignedPairs.get(selectedPairId).row.classList.remove("selected");
        }
        selectedPairId = id;
        const item = alignedPairs.get(id);
        if (!item) return;
        item.row.classList.add("selected");
        try { item.row.scrollIntoView({ block: "center", behavior: "smooth" }); } catch {}
        setTimeout(() => {
          if (selectedPairId === id && alignedPairs.has(id)) {
            alignedPairs.get(id).row.classList.remove("selected");
            selectedPairId = null;
          }
        }, 3500);
      }

      function formatDuration(seconds) {
        if (seconds < 60) return `${seconds}s`;
        const m = Math.round(seconds / 60);
        return `${m}m`;
      }

      function renderCatchup(msg) {
        if (!catchupDiv) return;
        catchupDiv.textContent = "";

        const meta = document.createElement("div");
        meta.className = "catchup-meta";
        meta.textContent = `対象: 直近 ${formatDuration(msg.window_seconds || 0)}（参照IDを押すと該当箇所へジャンプ）`;
        catchupDiv.appendChild(meta);

        const ul = document.createElement("ul");

        function addBullet(text, ids) {
          const li = document.createElement("li");
          const t = document.createElement("div");
          t.textContent = text;
          li.appendChild(t);
          if (ids && ids.length) {
            const refs = document.createElement("div");
            refs.className = "catchup-refs";
            for (const id of ids) {
              const chip = document.createElement("span");
              chip.className = "ref-chip";
              chip.textContent = id;
              chip.onclick = () => highlightPair(id);
              refs.appendChild(chip);
            }
            li.appendChild(refs);
          }
          return li;
        }

        const topic = (msg.topic || "").trim();
        ul.appendChild(addBullet(`今の話題: ${topic || "（不明）"}`, []));

        const important = Array.isArray(msg.important_points) ? msg.important_points : [];
        if (important.length) {
          const li = document.createElement("li");
          li.appendChild(document.createTextNode("重要点:"));
          const sub = document.createElement("ul");
          for (const p of important) {
            sub.appendChild(addBullet(p.text || "", p.ids || []));
          }
          li.appendChild(sub);
          ul.appendChild(li);
        } else {
          ul.appendChild(addBullet("重要点: （なし）", []));
        }

        const decisions = Array.isArray(msg.decisions) ? msg.decisions : [];
        if (decisions.length) {
          const li = document.createElement("li");
          li.appendChild(document.createTextNode("決定事項:"));
          const sub = document.createElement("ul");
          for (const d of decisions) {
            sub.appendChild(addBullet(d.text || "", d.ids || []));
          }
          li.appendChild(sub);
          ul.appendChild(li);
        } else {
          ul.appendChild(addBullet("決定事項: （なし）", []));
        }

        const nextTopic = (msg.next_topic || "").trim();
        ul.appendChild(addBullet(`次の話題: ${nextTopic || "（不明）"}`, []));

        catchupDiv.appendChild(ul);
      }

      function requestCatchup(seconds) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        try {
          catchupDiv.textContent = `Catch up 生成中…（直近 ${formatDuration(seconds)}）`;
          ws.send(JSON.stringify({ type: "catchup", seconds }));
        } catch {}
      }

	  function appendFinal(text) {
	    if (finalText && !finalText.endsWith("\\n")) finalText += " ";
	    finalText += text.trim();
	    partialText = "";
    updateView();
  }

  function setPartial(text) {
    partialText = text ? (text.trim() + " ") : "";
    updateView();
  }

  function appendTranslation(text) {
    finalTranslation += text;
    partialTranslation = "";
    updateTranslationView();
  }

  // リングバッファに書き込み
  function ringBufferWrite(samples) {
    for (let i = 0; i < samples.length; i++) {
      ringBuffer[ringWriteIndex] = samples[i];
      ringWriteIndex = (ringWriteIndex + 1) % RING_BUFFER_SIZE;
      // オーバーフロー時は古いデータを上書き（リアルタイム性優先）
      if (ringWriteIndex === ringReadIndex) {
        ringReadIndex = (ringReadIndex + 1) % RING_BUFFER_SIZE;
      }
    }
  }

  // リングバッファから読み出し可能なサンプル数
  function ringBufferAvailable() {
    if (ringWriteIndex >= ringReadIndex) {
      return ringWriteIndex - ringReadIndex;
    }
    return RING_BUFFER_SIZE - ringReadIndex + ringWriteIndex;
  }

  // リングバッファからフレームを送信（コピーなしでDataViewに直接書き込み）
  function flushFrames() {
    while (ringBufferAvailable() >= FRAME_SAMPLES) {
      for (let i = 0; i < FRAME_SAMPLES; i++) {
        frameView.setInt16(i * 2, ringBuffer[ringReadIndex], true);
        ringReadIndex = (ringReadIndex + 1) % RING_BUFFER_SIZE;
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(frameBuffer);
      }
    }
  }

  // Float32 -> Int16LE 変換（最適化版：事前確保配列使用）
  function toInt16LEOptimized(float32Array, outputArray) {
    const len = float32Array.length;
    for (let i = 0; i < len; i++) {
      let s = float32Array[i];
      s = s < -1 ? -1 : (s > 1 ? 1 : s);
      outputArray[i] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7FFF) | 0;
    }
    return len;
  }

  // ダウンサンプリング（最適化版）
  function downsampleOptimized(input, inputSampleRate, outputSampleRate, outputArray) {
    if (outputSampleRate === inputSampleRate) {
      for (let i = 0; i < input.length; i++) outputArray[i] = input[i];
      return input.length;
    }
    const ratio = inputSampleRate / outputSampleRate;
    const newLen = Math.round(input.length / ratio);
    let offset = 0;
    for (let i = 0; i < newLen; i++) {
      const nextOffset = Math.round((i + 1) * ratio);
      let acc = 0, count = 0;
      for (let j = offset; j < nextOffset && j < input.length; j++) { acc += input[j]; count++; }
      outputArray[i] = count ? (acc / count) : 0;
      offset = nextOffset;
    }
    return newLen;
  }

  // AudioWorkletプロセッサのコード
  const workletCode = `
    class PCMProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.buffer = new Float32Array(4096);
        this.bufferIndex = 0;
        this.targetSampleRate = 16000;
        this.chunkSize = 1024; // 送信チャンクサイズ
      }

      process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;

        const channelData = input[0];
        if (!channelData) return true;

        // バッファに追加
        for (let i = 0; i < channelData.length; i++) {
          this.buffer[this.bufferIndex++] = channelData[i];

          // バッファがいっぱいになったらメインスレッドに送信
          if (this.bufferIndex >= this.chunkSize) {
            this.port.postMessage({
              audioData: this.buffer.slice(0, this.bufferIndex),
              sampleRate: sampleRate
            });
            this.bufferIndex = 0;
          }
        }
        return true;
      }
    }
    registerProcessor('pcm-processor', PCMProcessor);
  `;

  // AudioWorkletの初期化
  async function initAudioWorklet(audioCtx, mediaStream) {
    try {
      const blob = new Blob([workletCode], { type: 'application/javascript' });
      const url = URL.createObjectURL(blob);
      await audioCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      workletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');

      // ワーカー用の一時バッファ（再割り当て防止）
      const tempFloat = new Float32Array(4096);
      const tempInt16 = new Int16Array(4096);

      workletNode.port.onmessage = (e) => {
        const { audioData, sampleRate } = e.data;
        // ダウンサンプリング
        const downLen = downsampleOptimized(audioData, sampleRate, TARGET_SR, tempFloat);
        // Int16変換
        toInt16LEOptimized(tempFloat.subarray(0, downLen), tempInt16);
        // リングバッファに書き込み
        ringBufferWrite(tempInt16.subarray(0, downLen));
        // フレーム送信
        flushFrames();
      };

      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      sourceNode.connect(workletNode);

      // 一部のブラウザでは destination 接続がないと process() が呼ばれないため、
      // GainNode で無音化して destination に接続（process() の確実な実行を保証）
      const gainNode = audioCtx.createGain();
      gainNode.gain.value = 0; // 無音化
      workletNode.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      useAudioWorklet = true;
      console.log('[Audio] Using AudioWorklet (high performance)');
      return true;
    } catch (err) {
      console.warn('[AudioWorklet] Not available, falling back to ScriptProcessor:', err);
      return false;
    }
  }

  // レガシーScriptProcessorNode（フォールバック）
  function initLegacyProcessor(audioCtx, mediaStream) {
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    legacyProcessor = audioCtx.createScriptProcessor(2048, 1, 1);

    // フォールバック用の一時バッファ
    const tempFloat = new Float32Array(4096);
    const tempInt16 = new Int16Array(4096);

    legacyProcessor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const downLen = downsampleOptimized(input, audioCtx.sampleRate, TARGET_SR, tempFloat);
      toInt16LEOptimized(tempFloat.subarray(0, downLen), tempInt16);
      ringBufferWrite(tempInt16.subarray(0, downLen));
      flushFrames();
    };

    sourceNode.connect(legacyProcessor);
    legacyProcessor.connect(audioCtx.destination);

    useAudioWorklet = false;
    console.log('[Audio] Using ScriptProcessorNode (legacy fallback)');
  }

		  async function start() {
	        stopping = false;
		    // リングバッファをリセット
		    ringWriteIndex = 0;
		    ringReadIndex = 0;
	
	    const scheme = (location.protocol === "https:") ? "wss" : "ws";
	    ws = new WebSocket(`${scheme}://${location.host}/ws`);
	    const wsLocal = ws;
	
		    wsLocal.onopen = async () => {
		      if (ws !== wsLocal) return;
		      setStatus("Connected. Capturing audio…");
		      updateIndicator("ws", "active", "Connected");
		      updateIndicator("audio", "", "Starting...");
		      updateIndicator("aws", "", "Connecting...");
	          updateIndicator("translation", "", "Idle");
	          updateIndicator("catchup", "", "Idle");

      // Wake Lock
      try {
        if ('wakeLock' in navigator) {
          wakeLock = await navigator.wakeLock.request('screen');
          console.log('[Wake Lock] Screen wake lock acquired');
          wakeLock.addEventListener('release', () => {
            console.log('[Wake Lock] Screen wake lock released');
          });
        }
      } catch (err) {
        console.warn('[Wake Lock] Failed:', err);
      }

	      // キープアライブ
	      pingInterval = setInterval(() => {
	        if (wsLocal && wsLocal.readyState === WebSocket.OPEN) {
	          wsLocal.send(JSON.stringify({type: "pong"}));
	        }
	      }, 25000);
	    };
	
		    wsLocal.onclose = () => {
		      if (ws !== wsLocal) return;
		      setStatus("Disconnected.");
		      updateIndicator("ws", "", "Disconnected");
		      updateIndicator("audio", "", "Idle");
		      updateIndicator("aws", "", "Idle");
	          updateIndicator("translation", "", "Idle");
	          updateIndicator("catchup", "", "Idle");
		      if (pingInterval) clearInterval(pingInterval);
		    };
	
		    wsLocal.onerror = () => {
		      if (ws !== wsLocal) return;
		      setStatus("WebSocket error.");
		      updateIndicator("ws", "error", "Error");
	          updateIndicator("translation", "", "Idle");
	          updateIndicator("catchup", "", "Idle");
		      if (pingInterval) clearInterval(pingInterval);
		    };
	
		    wsLocal.onmessage = (event) => {
		      if (ws !== wsLocal) return;
		      try {
		        const msg = JSON.parse(event.data);
		        if (msg.type === "final") {
		          appendFinal(msg.text);
		        } else if (msg.type === "speculative" || msg.type === "partial") {
	          setPartial(msg.text);
	        } else if (msg.type === "translation") {
	          appendTranslation(msg.text);
            } else if (msg.type === "aligned_en") {
              addAlignedPair(msg.id, msg.en);
            } else if (msg.type === "aligned_ja") {
              setAlignedJa(msg.id, msg.ja);
	            } else if (msg.type === "catchup_result") {
	              renderCatchup(msg);
                  updateIndicator("catchup", "active", "Ready");
	            } else if (msg.type === "catchup_error") {
	              catchupDiv.textContent = "Catch up Error: " + (msg.error || "Unknown");
                  updateIndicator("catchup", "error", "Error");
		        } else if (msg.type === "translation_error") {
		          setStatus("Translation Error: " + (msg.error || "Unknown"));
                  updateIndicator("translation", "error", "Error");
		        } else if (msg.type === "info") {
		          setStatus(msg.text);
		        } else if (msg.type === "status") {
	          if (msg.status === "aws_connected") {
	            updateIndicator("aws", "active", "Connected");
	          } else if (msg.status === "aws_error") {
	            updateIndicator("aws", "error", "Error");
	            setStatus("AWS Error: " + (msg.error || "Unknown"));
	          } else if (msg.status === "audio_receiving") {
	            updateIndicator("audio", "active", "Receiving");
	          } else if (msg.status === "transcribing") {
	            updateIndicator("aws", "active", "Transcribing");
              } else if (msg.status === "translation_translating") {
                updateIndicator("translation", "active", "Translating");
              } else if (msg.status === "translation_throttled") {
                updateIndicator("translation", "active", "Throttled");
              } else if (msg.status === "translation_disabled") {
                updateIndicator("translation", "error", "Disabled");
              } else if (msg.status === "translation_idle") {
                updateIndicator("translation", "", "Idle");
              } else if (msg.status === "catchup_generating") {
                updateIndicator("catchup", "active", "Generating");
              } else if (msg.status === "catchup_ready") {
                updateIndicator("catchup", "active", "Ready");
              } else if (msg.status === "catchup_throttled") {
                updateIndicator("catchup", "active", "Throttled");
              } else if (msg.status === "catchup_error") {
                updateIndicator("catchup", "error", "Error");
              } else if (msg.status === "catchup_idle") {
                updateIndicator("catchup", "", "Idle");
		          }
		        } else if (msg.type === "ping") {
		          wsLocal.send(JSON.stringify({type: "pong"}));
		        } else {
		          setPartial(msg.text || "");
	        }
	      } catch (e) {
        appendFinal(event.data);
      }
    };

        const inputSource = (inputSourceSel && inputSourceSel.value) ? inputSourceSel.value : "mic";
        let mediaStream;
        if (inputSource === "tab") {
          if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
            throw new Error("Tab Audio is not supported in this browser. Use Chrome/Edge.");
          }
          setStatus("Select a browser tab and enable 'Share tab audio'…");
          mediaStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
          const audioTracks = mediaStream.getAudioTracks();
          if (!audioTracks || audioTracks.length === 0) {
            // Common pitfall: the user shared a window/screen without tab audio.
            mediaStream.getTracks().forEach(t => { try { t.stop(); } catch {} });
            throw new Error("No audio track captured. In the share dialog, choose a TAB and enable 'Share tab audio'.");
          }
        } else {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
          });
        }

        captureStream = mediaStream;
        // If the user stops sharing (tab audio), end the session gracefully.
        captureStream.getTracks().forEach((t) => {
          try { t.addEventListener("ended", () => stop()); } catch {}
        });

	    audioContext = new (window.AudioContext || window.webkitAudioContext)();

	    // AudioWorkletを試行、失敗時はScriptProcessorにフォールバック
	    const workletSuccess = await initAudioWorklet(audioContext, mediaStream);
    if (!workletSuccess) {
      initLegacyProcessor(audioContext, mediaStream);
    }

    updateIndicator("audio", "active", "Capturing");

		    startBtn.disabled = true;
		    stopBtn.disabled = false;
		    downloadBtn.disabled = false;
		    clearBtn.disabled = false;
        catchup30Btn.disabled = false;
        catchup120Btn.disabled = false;
        catchup300Btn.disabled = false;
        if (inputSourceSel) inputSourceSel.disabled = true;
		  }

	  function stop() {
        if (stopping) return;
        stopping = true;
	    setStatus("Stopping…");
	    if (ws) { try { ws.close(); } catch {} }
	    ws = null;

	    if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
    if (workletNode) { try { workletNode.disconnect(); } catch {} }
    if (legacyProcessor) { try { legacyProcessor.disconnect(); } catch {} }
    if (sourceNode) { try { sourceNode.disconnect(); } catch {} }
	    if (audioContext) { try { audioContext.close(); } catch {} }
        if (captureStream) {
          try { captureStream.getTracks().forEach(t => t.stop()); } catch {}
          captureStream = null;
        }

    // Wake Lock解放
    if (wakeLock) {
      try {
        wakeLock.release();
        console.log('[Wake Lock] Released');
      } catch (err) {
        console.warn('[Wake Lock] Failed to release:', err);
      }
      wakeLock = null;
    }

    workletNode = null;
    legacyProcessor = null;
    sourceNode = null;
    audioContext = null;

    // リングバッファをリセット
    ringWriteIndex = 0;
    ringReadIndex = 0;

    updateIndicator("ws", "", "Disconnected");
    updateIndicator("audio", "", "Idle");
    updateIndicator("aws", "", "Idle");

	    startBtn.disabled = false;
	    stopBtn.disabled = true;
        catchup30Btn.disabled = true;
        catchup120Btn.disabled = true;
        catchup300Btn.disabled = true;
        if (inputSourceSel) inputSourceSel.disabled = false;
	  }

  function download() {
    const transcriptContent = (finalText + partialText).trim();
    const translationContent = (finalTranslation + partialTranslation).trim();
    const content = [
      "=== English Transcript ===",
      transcriptContent || "…",
      "",
      "=== Japanese Translation ===",
      translationContent || "…",
      "",
    ].join("\\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `transcript_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

	  function clearAll() {
	    finalText = "";
	    partialText = "";
	    updateView();
	    finalTranslation = "";
	    partialTranslation = "";
	    updateTranslationView();
        alignedPairs.clear();
        if (pairsDiv) pairsDiv.textContent = "…";
        if (catchupDiv) catchupDiv.textContent = "Press a button (30s / 2m / 5m) to catch up.";
	    if (ws && ws.readyState === WebSocket.OPEN) {
	      try { ws.send(JSON.stringify({type: "clear"})); } catch {}
	    }
	  }

		  startBtn.onclick = () => start().catch(err => {
		    stop();
		    setStatus(String(err));
		    updateIndicator("audio", "error", "Error");
		  });
		  stopBtn.onclick = () => stop();
		  downloadBtn.onclick = () => download();
		  clearBtn.onclick = () => clearAll();
	      catchup30Btn.onclick = () => requestCatchup(30);
	      catchup120Btn.onclick = () => requestCatchup(120);
      catchup300Btn.onclick = () => requestCatchup(300);
})();
</script>
</body>
</html>"""
