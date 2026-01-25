import asyncio
from collections import deque
import json
import logging
import os
import re
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from aws_sdk_bedrock_runtime.models import (
    ResourceNotFoundException,
    ThrottlingException,
    ValidationException,
)

from .bedrock import NovaSonicSession, get_bedrock_client, stream_prompt_text_deltas
from .logging_setup import setup_logging
from .parsing import (
    _extract_first_json_object,
    _extract_generation_stage,
    _is_interrupted_tag,
    _parse_indexed_output_line,
)
from .settings import (
    AWS_REGION_DEFAULT,
    CATCHUP_LOG_MAX_ITEMS,
    CATCHUP_LOG_MAX_SECONDS,
    CATCHUP_MAX_INPUT_CHARS,
    CATCHUP_MAX_TOKENS,
    CATCHUP_MIN_INTERVAL_SECONDS,
    CATCHUP_MODEL_ID,
    CATCHUP_MODEL_ID_FALLBACK,
    KEEPALIVE_SECONDS,
    MEETING_ASSIST_MAX_INPUT_CHARS,
    MEETING_ASSIST_MAX_TOKENS,
    MEETING_ASSIST_MIN_INTERVAL_SECONDS,
    MEETING_ASSIST_MODEL_ID,
    MEETING_ASSIST_MODEL_ID_FALLBACK,
    MEETING_ASSIST_WINDOW_SECONDS,
    MODEL_ID,
    RENEW_SECONDS,
    TRANSLATION_DEBOUNCE_SECONDS,
    TRANSLATION_MAX_BATCH_CHARS,
    TRANSLATION_MAX_TOKENS,
    TRANSLATION_MIN_INTERVAL_SECONDS,
    TRANSLATION_MODEL_ID,
    TRANSLATION_MODEL_ID_FALLBACK,
)
from .ui import INDEX_HTML

setup_logging()
logger = logging.getLogger("main")

app = FastAPI()

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
    meeting_assist_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)

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

    meeting_assist_enabled = False
    meeting_assist_count = 0
    meeting_assist_model_id_in_use = MEETING_ASSIST_MODEL_ID
    meeting_assist_model_id_fallback = MEETING_ASSIST_MODEL_ID_FALLBACK
    meeting_assist_last_request_at = 0.0
    meeting_assist_last_seq_scheduled = 0

    logger.info(
        "Translation model configured: "
        f"primary='{translation_model_id_in_use}', fallback='{translation_model_id_fallback}'"
    )
    logger.info(
        "Catch-up model configured: "
        f"primary='{catchup_model_id_in_use}', fallback='{catchup_model_id_fallback}'"
    )
    logger.info(
        "Meeting assist model configured: "
        f"primary='{meeting_assist_model_id_in_use}', fallback='{meeting_assist_model_id_fallback}'"
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

    def _schedule_meeting_assist(force: bool = False) -> None:
        nonlocal meeting_assist_last_request_at, meeting_assist_last_seq_scheduled
        if not meeting_assist_enabled:
            return
        if not force and not utterance_log:
            return
        if meeting_assist_q.full():
            return
        now_mono = time.monotonic()
        if not force:
            if translation_segment_seq == meeting_assist_last_seq_scheduled:
                return
            if (now_mono - meeting_assist_last_request_at) < MEETING_ASSIST_MIN_INTERVAL_SECONDS:
                return
        meeting_assist_last_request_at = now_mono
        meeting_assist_last_seq_scheduled = translation_segment_seq
        req = {"epoch": conversation_epoch, "window_seconds": MEETING_ASSIST_WINDOW_SECONDS}
        try:
            meeting_assist_q.put_nowait(req)
        except asyncio.QueueFull:
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
                        client,
                        prompt=prompt,
                        model_id=translation_model_id_in_use,
                        max_tokens=TRANSLATION_MAX_TOKENS,
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
                    client,
                    prompt=prompt,
                    model_id=catchup_model_id_in_use,
                    max_tokens=CATCHUP_MAX_TOKENS,
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

    async def meeting_assist_worker():
        nonlocal meeting_assist_count, meeting_assist_model_id_in_use, conversation_epoch, meeting_assist_enabled
        logger.info("meeting_assist_worker task started, waiting for meeting assist requests...")
        def _parse_json_maybe(text: str) -> Optional[dict]:
            if not text:
                return None
            json_part = _extract_first_json_object(text)
            if not json_part:
                return None
            try:
                return json.loads(json_part)
            except json.JSONDecodeError:
                # 軽い修正: 末尾の余分なカンマを除去して再試行
                cleaned = re.sub(r",\s*([}\]])", r"\1", json_part)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None

        while not stop_event.is_set():
            try:
                req = await meeting_assist_q.get()
            except asyncio.CancelledError:
                break

            if stop_event.is_set():
                break
            if not meeting_assist_enabled:
                continue
            if req.get("epoch") != conversation_epoch:
                continue

            window_seconds = int(req.get("window_seconds") or MEETING_ASSIST_WINDOW_SECONDS)
            window_seconds = max(30, min(window_seconds, CATCHUP_LOG_MAX_SECONDS))
            batch_epoch = req.get("epoch")

            if not await safe_send({"type": "status", "status": "meeting_assist_generating"}):
                return

            now_ts = time.time()
            cutoff = now_ts - window_seconds
            _prune_utterance_log(now_ts)

            window_items = [u for u in utterance_log if float(u.get("ts", 0)) >= cutoff]
            if not window_items:
                msg = {
                    "type": "meeting_assist_result",
                    "window_seconds": window_seconds,
                    "topic": "",
                    "participants": [],
                    "direction": "",
                    "options": [],
                    "english_advice": [],
                }
                if meeting_assist_enabled and batch_epoch == conversation_epoch:
                    if not await safe_send(msg):
                        return
                    await safe_send({"type": "status", "status": "meeting_assist_ready"})
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
                if total_chars + len(block) > MEETING_ASSIST_MAX_INPUT_CHARS and chosen:
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

            meeting_assist_count += 1
            logger.info(
                "Meeting assist request #"
                f"{meeting_assist_count}: window={window_seconds}s, segments={len(chosen)}, chars={len(transcript_text)}"
            )

            prompt = (
                "You are a real-time meeting navigator assistant.\n"
                "A participant wants to understand the meeting status and how to respond.\n\n"
                "You will receive transcript segments with IDs like u12.\n"
                "Each segment may have EN and sometimes JA.\n"
                "Use JA when available; otherwise use EN.\n\n"
                "Return ONLY valid JSON (no markdown), with this schema:\n"
                "{\n"
                '  "topic": string,\n'
                '  "participants": [{"name": string, "role": string, "notes": string}],\n'
                '  "direction": string,\n'
                '  "options": [{"text": string}],\n'
                '  "english_advice": [{"ja": string, "en": string}]\n'
                "}\n\n"
                "Rules:\n"
                "- All strings must be Japanese except english_advice[].en (English).\n"
                "- Participants are inferred from the transcript; if unsure, keep fields empty or note '推定'.\n"
                "- If there are no options or advice, return empty arrays.\n"
                "- Keep it concise and practical for a live meeting.\n"
                "- Do not invent facts not present in the transcript.\n\n"
                f"Time window: last {window_seconds} seconds.\n\n"
                "Transcript:\n"
                f"{transcript_text}\n"
            )

            try:
                buf = ""
                async for delta in stream_prompt_text_deltas(
                    client,
                    prompt=prompt,
                    model_id=meeting_assist_model_id_in_use,
                    max_tokens=MEETING_ASSIST_MAX_TOKENS,
                ):
                    if batch_epoch != conversation_epoch:
                        buf = ""
                        break
                    buf += delta

                if batch_epoch != conversation_epoch or not buf.strip() or not meeting_assist_enabled:
                    continue

                parsed = _parse_json_maybe(buf)
                if not isinstance(parsed, dict):
                    # 1回だけ再試行：モデル出力を修復させる
                    fix_prompt = (
                        "You will be given a response that should be JSON but is invalid.\n"
                        "Repair it into valid JSON only (no markdown), parsable by json.loads.\n"
                        "Use double quotes for all keys/strings. No trailing commas.\n"
                        "If a field is missing, use empty string or empty array.\n\n"
                        "Schema:\n"
                        "{\n"
                        '  "topic": string,\n'
                        '  "participants": [{"name": string, "role": string, "notes": string}],\n'
                        '  "direction": string,\n'
                        '  "options": [{"text": string}],\n'
                        '  "english_advice": [{"ja": string, "en": string}]\n'
                        "}\n\n"
                        "Invalid output:\n"
                        f"{buf}\n"
                    )
                    buf_retry = ""
                    async for delta in stream_prompt_text_deltas(
                        client,
                        prompt=fix_prompt,
                        model_id=meeting_assist_model_id_in_use,
                        max_tokens=MEETING_ASSIST_MAX_TOKENS,
                    ):
                        if batch_epoch != conversation_epoch:
                            buf_retry = ""
                            break
                        buf_retry += delta
                    parsed = _parse_json_maybe(buf_retry)

                if not isinstance(parsed, dict):
                    raise ValueError("Invalid JSON from model")

                topic = str(parsed.get("topic") or "").strip()
                direction = str(parsed.get("direction") or "").strip()

                def _norm_participants(items):
                    out = []
                    if not isinstance(items, list):
                        return out
                    for it in items:
                        if isinstance(it, str):
                            text = it.strip()
                            if text:
                                out.append({"name": text, "role": "", "notes": ""})
                            continue
                        if not isinstance(it, dict):
                            continue
                        name = str(it.get("name") or "").strip()
                        role = str(it.get("role") or "").strip()
                        notes = str(it.get("notes") or "").strip()
                        if name or role or notes:
                            out.append({"name": name, "role": role, "notes": notes})
                    return out[:8]

                def _norm_options(items):
                    out = []
                    if not isinstance(items, list):
                        return out
                    for it in items:
                        if isinstance(it, str):
                            text = it.strip()
                            if text:
                                out.append({"text": text})
                            continue
                        if not isinstance(it, dict):
                            continue
                        text = str(
                            it.get("text")
                            or it.get("option")
                            or it.get("summary")
                            or ""
                        ).strip()
                        if text:
                            out.append({"text": text})
                    return out[:6]

                def _norm_advice(items):
                    out = []
                    if not isinstance(items, list):
                        return out
                    for it in items:
                        if isinstance(it, str):
                            text = it.strip()
                            if text:
                                out.append({"ja": text, "en": ""})
                            continue
                        if not isinstance(it, dict):
                            continue
                        ja = str(it.get("ja") or it.get("intent") or it.get("note") or "").strip()
                        en = str(
                            it.get("en") or it.get("phrase") or it.get("say") or ""
                        ).strip()
                        if ja or en:
                            out.append({"ja": ja, "en": en})
                    return out[:6]

                msg = {
                    "type": "meeting_assist_result",
                    "window_seconds": window_seconds,
                    "topic": topic,
                    "participants": _norm_participants(parsed.get("participants") or []),
                    "direction": direction,
                    "options": _norm_options(parsed.get("options") or []),
                    "english_advice": _norm_advice(parsed.get("english_advice") or []),
                }
                if not meeting_assist_enabled or batch_epoch != conversation_epoch:
                    continue
                if not await safe_send(msg):
                    return
                await safe_send({"type": "status", "status": "meeting_assist_ready"})

            except ValidationException as e:
                message_raw = getattr(e, "message", None) or str(e) or ""
                message = message_raw.lower()
                if "model identifier is invalid" in message:
                    meeting_assist_enabled = False
                    try:
                        while True:
                            meeting_assist_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await safe_send(
                        {
                            "type": "meeting_assist_error",
                            "error": (
                                "Meeting Assist のモデルIDが無効です。"
                                " `.env` の MEETING_ASSIST_MODEL_ID を有効なID/ARNに設定してください。"
                            ),
                        }
                    )
                    await safe_send({"type": "status", "status": "meeting_assist_idle"})
                    continue
                if (
                    "on-demand throughput" in message
                    and "inference profile" in message
                    and meeting_assist_model_id_in_use != meeting_assist_model_id_fallback
                ):
                    logger.warning(
                        "Meeting assist model does not support on-demand throughput. "
                        f"Switching to inference profile: {meeting_assist_model_id_fallback}"
                    )
                    meeting_assist_model_id_in_use = meeting_assist_model_id_fallback
                    try:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "info",
                                    "text": (
                                        "Meeting Assist モデルを inference profile に切り替えました: "
                                        f"model_id={meeting_assist_model_id_in_use}"
                                    ),
                                }
                            )
                        )
                    except Exception:
                        pass
                    continue

                logger.error(f"Meeting assist error: {e}", exc_info=True)
                await safe_send({"type": "status", "status": "meeting_assist_error"})
                await safe_send({"type": "meeting_assist_error", "error": str(e)})

            except ThrottlingException as e:
                logger.warning(f"Meeting assist throttled: {e}")
                await safe_send({"type": "status", "status": "meeting_assist_throttled"})
                await safe_send(
                    {
                        "type": "meeting_assist_error",
                        "error": "混雑しています。少し待ってから再試行してください。",
                    }
                )

            except Exception as e:
                logger.error(f"Meeting assist error: {e}", exc_info=True)
                await safe_send({"type": "status", "status": "meeting_assist_error"})
                await safe_send({"type": "meeting_assist_error", "error": str(e)})

        logger.info("meeting_assist_worker task ended.")

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
                                _schedule_meeting_assist()
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
                                    _schedule_meeting_assist()
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

            except asyncio.CancelledError:
                break
            except ValidationException as e:
                message = getattr(e, "message", None) or str(e) or ""
                if "Timed out waiting for input events" in message:
                    logger.info("Bedrock output timed out waiting for input events (likely no audio).")
                    break
                logger.error(f"Error reading AWS output: {e}", exc_info=True)
                await safe_send({"type": "status", "status": "aws_error", "error": str(e)})
                break
            except Exception as e:
                if stop_event.is_set():
                    break
                logger.error(f"Error reading AWS output: {e}", exc_info=True)
                await safe_send({"type": "status", "status": "aws_error", "error": str(e)})
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
            if not await safe_send(
                {"type": "info", "text": "Session renewed to avoid the 8-minute limit."}
            ):
                stop_event.set()
                return

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
            if not await safe_send({"type": "ping"}):
                logger.warning("Keepalive ping failed: WebSocket send failed")
                stop_event.set()
                return
            logger.debug(f"Sent keepalive ping (audio frames: {audio_frame_count})")

    async def recv_from_browser():
        nonlocal audio_frame_count, last_audio_time, translation_started, translation_segment_seq, conversation_epoch
        nonlocal meeting_assist_enabled, meeting_assist_last_request_at, meeting_assist_last_seq_scheduled
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
                            meeting_assist_last_request_at = 0.0
                            meeting_assist_last_seq_scheduled = 0
                            utterance_log.clear()
                            utterance_by_id.clear()
                            if not await safe_send(
                                {"type": "status", "status": "translation_idle"}
                            ):
                                stop_event.set()
                                return
                            if not await safe_send({"type": "status", "status": "catchup_idle"}):
                                stop_event.set()
                                return
                            if meeting_assist_enabled:
                                if not await safe_send(
                                    {"type": "status", "status": "meeting_assist_enabled"}
                                ):
                                    stop_event.set()
                                    return
                            else:
                                if not await safe_send(
                                    {"type": "status", "status": "meeting_assist_idle"}
                                ):
                                    stop_event.set()
                                    return
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
                            try:
                                while True:
                                    meeting_assist_q.get_nowait()
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
                        elif msg.get("type") == "meeting_assist_toggle":
                            enabled = bool(msg.get("enabled"))
                            meeting_assist_enabled = enabled
                            meeting_assist_last_request_at = 0.0
                            meeting_assist_last_seq_scheduled = 0
                            if not enabled:
                                try:
                                    while True:
                                        meeting_assist_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                await safe_send({"type": "status", "status": "meeting_assist_idle"})
                            else:
                                await safe_send({"type": "status", "status": "meeting_assist_enabled"})
                                _schedule_meeting_assist(force=True)
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
                            if not await safe_send({"type": "status", "status": "audio_receiving"}):
                                stop_event.set()
                                return
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
                    if not await safe_send(
                        {"type": "status", "status": "aws_error", "error": str(e)}
                    ):
                        stop_event.set()
                        return

    async with session_lock:
        await start_session()

    tasks = [
        asyncio.create_task(renew_loop()),
        asyncio.create_task(keepalive_loop()),
        asyncio.create_task(recv_from_browser()),
        asyncio.create_task(send_to_bedrock()),
        asyncio.create_task(translation_worker()),
        asyncio.create_task(catchup_worker()),
        asyncio.create_task(meeting_assist_worker()),
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
    return INDEX_HTML
