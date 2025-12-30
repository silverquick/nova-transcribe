import asyncio
import base64
import json
import logging
import os
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
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.identity import EnvironmentCredentialsResolver

# ログ設定
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
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
AWS_REGION_DEFAULT = "ap-northeast-1"

# Bedrock側ストリーム寿命(8分)手前で更新
RENEW_SECONDS = 7 * 60 + 45  # 7m45s
# プロキシタイムアウト回避用キープアライブ間隔
KEEPALIVE_SECONDS = 30


def get_bedrock_client(region: str) -> BedrockRuntimeClient:
    cfg = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
    )
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


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # .env から region を渡す想定（未設定ならデフォルト）
    region = os.getenv("AWS_REGION", AWS_REGION_DEFAULT)

    client = get_bedrock_client(region)
    audio_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=256)

    # シンプルなプロンプトで低遅延化
    system_prompt = "Transcribe the user's speech into text accurately."

    session_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    session: Optional[NovaSonicSession] = None
    output_task: Optional[asyncio.Task] = None

    # 統計情報
    audio_frame_count = 0
    last_audio_time = time.time()
    transcription_count = 0

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
        nonlocal session, transcription_count
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

                    # USERロールとASSISTANTロールの両方を表示（Nova 2 Sonicは会話型AIのため）
                    if session.current_role in ["USER", "ASSISTANT"]:
                        stage_lower = (session.current_generation_stage or "FINAL").lower()
                        transcription_count += 1
                        logger.info(f"Transcription #{transcription_count} ({stage_lower}, {role}): {text[:50]}...")
                        await websocket.send_text(json.dumps({"type": stage_lower, "text": text}))
                        await websocket.send_text(json.dumps({"type": "status", "status": "transcribing"}))
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
            await asyncio.sleep(RENEW_SECONDS)
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
            await asyncio.sleep(KEEPALIVE_SECONDS)
            if stop_event.is_set():
                return
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
                logger.debug(f"Sent keepalive ping (audio frames: {audio_frame_count})")
            except Exception as e:
                logger.warning(f"Keepalive ping failed: {e}")
                break

    async def recv_from_browser():
        nonlocal audio_frame_count, last_audio_time
        try:
            while not stop_event.is_set():
                message = await websocket.receive()

                # テキストメッセージ（JSON）の処理
                if message["type"] == "websocket.receive" and "text" in message:
                    try:
                        msg = json.loads(message["text"])
                        if msg.get("type") == "pong":
                            logger.debug("Received pong from browser")
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
                        logger.warning("Audio queue full, dropping frame")

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
            async with session_lock:
                if session and session.is_active:
                    try:
                        await session.send_audio(frame)
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
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 20px; background: #f6f7f9; }
    h1 { margin: 0 0 6px 0; font-size: 20px; }
    .sub { color: #555; margin-bottom: 14px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; align-items: center; }
    button { padding: 10px 14px; font-size: 14px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #status { font-size: 13px; color: #333; }
    .status-indicators { display: flex; gap: 12px; margin: 10px 0; font-size: 13px; }
    .indicator { display: flex; align-items: center; gap: 6px; }
    .indicator-dot { width: 10px; height: 10px; border-radius: 50%; background: #ccc; }
    .indicator-dot.active { background: #22c55e; animation: pulse 2s infinite; }
    .indicator-dot.error { background: #ef4444; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    #transcript {
      white-space: pre-wrap;
      min-height: 360px;
      max-height: 500px;
      overflow-y: auto;
      background: #fff;
      border: 1px solid #d9dde3;
      padding: 14px;
      border-radius: 10px;
      font-size: 16px;
      line-height: 1.5;
    }
    .hint { margin-top: 10px; font-size: 12px; color: #666; }
    code { background: #eef1f5; padding: 1px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Amazon Nova 2 Sonic – Real-time Transcription</h1>
  <div class="sub">Language: <b>English</b> (audio input: 16 kHz, mono, 16-bit PCM)</div>

  <div class="controls">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
    <button id="download" disabled>Download TXT</button>
    <button id="clear" disabled>Clear</button>
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
  </div>

  <div id="transcript">Press "Start" and speak.</div>
  <div class="hint">
    If you deploy behind HTTPS, this page will automatically use <code>wss://</code>. Microphone access requires a secure context (HTTPS or localhost).
    <br>Status indicators show connection health in real-time.
  </div>

<script>
(() => {
  let ws = null;
  let audioContext = null;
  let sourceNode = null;
  let processor = null;
  let pingInterval = null;
  let wakeLock = null;

  let finalText = "";
  let partialText = "";

  const TARGET_SR = 16000;
  const FRAME_SAMPLES = 1600; // 100ms at 16kHz (最適なレスポンス速度とネットワーク効率のバランス)
  let pcmBuffer = new Int16Array(0);

  const transcriptDiv = document.getElementById("transcript");
  const statusDiv = document.getElementById("status");
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const downloadBtn = document.getElementById("download");
  const clearBtn = document.getElementById("clear");

  const wsIndicator = document.getElementById("ws-indicator");
  const wsStatus = document.getElementById("ws-status");
  const audioIndicator = document.getElementById("audio-indicator");
  const audioStatus = document.getElementById("audio-status");
  const awsIndicator = document.getElementById("aws-indicator");
  const awsStatus = document.getElementById("aws-status");

  function setStatus(msg) { statusDiv.textContent = msg || ""; }

  function updateIndicator(type, status, text) {
    let indicator, statusSpan;
    if (type === "ws") { indicator = wsIndicator; statusSpan = wsStatus; }
    else if (type === "audio") { indicator = audioIndicator; statusSpan = audioStatus; }
    else if (type === "aws") { indicator = awsIndicator; statusSpan = awsStatus; }

    indicator.className = "indicator-dot " + status;
    statusSpan.textContent = text;
  }

  function updateView() {
    const merged = finalText + (partialText ? partialText : "");
    transcriptDiv.textContent = merged || "…";
    // 自動スクロール（最新のテキストを表示）
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
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

  function toInt16LE(float32Array) {
    const out = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let s = Math.max(-1, Math.min(1, float32Array[i]));
      out[i] = (s < 0 ? s * 0x8000 : s * 0x7FFF);
    }
    return out;
  }

  function downsampleBuffer(input, inputSampleRate, outputSampleRate) {
    if (outputSampleRate === inputSampleRate) return input;
    if (outputSampleRate > inputSampleRate) throw new Error("Output sample rate must be <= input sample rate");
    const ratio = inputSampleRate / outputSampleRate;
    const newLen = Math.round(input.length / ratio);
    const result = new Float32Array(newLen);
    let offset = 0;
    for (let i = 0; i < newLen; i++) {
      const nextOffset = Math.round((i + 1) * ratio);
      let acc = 0, count = 0;
      for (let j = offset; j < nextOffset && j < input.length; j++) { acc += input[j]; count++; }
      result[i] = count ? (acc / count) : 0;
      offset = nextOffset;
    }
    return result;
  }

  function concatInt16(a, b) {
    const out = new Int16Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function int16ToArrayBufferLE(int16Arr) {
    const buf = new ArrayBuffer(int16Arr.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < int16Arr.length; i++) view.setInt16(i * 2, int16Arr[i], true);
    return buf;
  }

  function flushFrames() {
    while (pcmBuffer.length >= FRAME_SAMPLES) {
      const frame = pcmBuffer.subarray(0, FRAME_SAMPLES);
      pcmBuffer = pcmBuffer.subarray(FRAME_SAMPLES);
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(int16ToArrayBufferLE(frame));
    }
  }

  async function start() {
    const scheme = (location.protocol === "https:") ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/ws`);

    ws.onopen = async () => {
      setStatus("Connected. Capturing audio…");
      updateIndicator("ws", "active", "Connected");
      updateIndicator("audio", "", "Starting...");
      updateIndicator("aws", "", "Connecting...");

      // 画面スリープ防止（Wake Lock API）
      try {
        if ('wakeLock' in navigator) {
          wakeLock = await navigator.wakeLock.request('screen');
          console.log('[Wake Lock] Screen wake lock acquired');
          wakeLock.addEventListener('release', () => {
            console.log('[Wake Lock] Screen wake lock released');
          });
        }
      } catch (err) {
        console.warn('[Wake Lock] Failed to acquire wake lock:', err);
      }

      // クライアント側からのキープアライブ (サーバーからのpingに対してpongを返す)
      pingInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({type: "pong"}));
        }
      }, 25000);
    };

    ws.onclose = () => {
      setStatus("Disconnected.");
      updateIndicator("ws", "", "Disconnected");
      updateIndicator("audio", "", "Idle");
      updateIndicator("aws", "", "Idle");
      if (pingInterval) clearInterval(pingInterval);
    };

    ws.onerror = () => {
      setStatus("WebSocket error.");
      updateIndicator("ws", "error", "Error");
      if (pingInterval) clearInterval(pingInterval);
    };

    ws.onmessage = (event) => {
      console.log("[WS] Received:", event.data);
      try {
        const msg = JSON.parse(event.data);
        console.log("[WS] Parsed:", msg);

        if (msg.type === "final") {
          console.log("[WS] Final text:", msg.text);
          appendFinal(msg.text);
        } else if (msg.type === "speculative" || msg.type === "partial") {
          console.log("[WS] Partial text:", msg.text);
          setPartial(msg.text);
        } else if (msg.type === "info") {
          console.log("[WS] Info:", msg.text);
          setStatus(msg.text);
        } else if (msg.type === "status") {
          console.log("[WS] Status update:", msg.status);
          // サーバーからのステータス更新
          if (msg.status === "aws_connected") {
            updateIndicator("aws", "active", "Connected");
          } else if (msg.status === "aws_error") {
            updateIndicator("aws", "error", "Error");
            setStatus("AWS Error: " + (msg.error || "Unknown"));
          } else if (msg.status === "audio_receiving") {
            updateIndicator("audio", "active", "Receiving");
          } else if (msg.status === "transcribing") {
            updateIndicator("aws", "active", "Transcribing");
          }
        } else if (msg.type === "ping") {
          console.log("[WS] Ping received, sending pong");
          // サーバーからのpingには自動応答
          ws.send(JSON.stringify({type: "pong"}));
        } else {
          console.log("[WS] Unknown type:", msg.type);
          setPartial(msg.text || "");
        }
      } catch (e) {
        console.error("[WS] Parse error:", e, event.data);
        appendFinal(event.data);
      }
    };

    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(2048, 1, 1);

    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const down = downsampleBuffer(input, audioContext.sampleRate, TARGET_SR);
      const int16 = toInt16LE(down);
      pcmBuffer = concatInt16(pcmBuffer, int16);
      flushFrames();
    };

    sourceNode.connect(processor);
    processor.connect(audioContext.destination);

    updateIndicator("audio", "active", "Capturing");

    startBtn.disabled = true;
    stopBtn.disabled = false;
    downloadBtn.disabled = false;
    clearBtn.disabled = false;
  }

  function stop() {
    setStatus("Stopping…");
    if (ws) { try { ws.close(); } catch {} }
    ws = null;
    pcmBuffer = new Int16Array(0);

    if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
    if (processor) { try { processor.disconnect(); } catch {} }
    if (sourceNode) { try { sourceNode.disconnect(); } catch {} }
    if (audioContext) { try { audioContext.close(); } catch {} }

    // Wake Lock解放
    if (wakeLock) {
      try {
        wakeLock.release();
        console.log('[Wake Lock] Released wake lock');
      } catch (err) {
        console.warn('[Wake Lock] Failed to release wake lock:', err);
      }
      wakeLock = null;
    }

    processor = null;
    sourceNode = null;
    audioContext = null;

    updateIndicator("ws", "", "Disconnected");
    updateIndicator("audio", "", "Idle");
    updateIndicator("aws", "", "Idle");

    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  function download() {
    const content = (finalText + partialText).trim();
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
  }

  startBtn.onclick = () => start().catch(err => setStatus(String(err)));
  stopBtn.onclick = () => stop();
  downloadBtn.onclick = () => download();
  clearBtn.onclick = () => clearAll();
})();
</script>
</body>
</html>"""
