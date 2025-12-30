# Amazon Nova 2 Sonic リアルタイム英語文字起こし Webアプリ（.env + `uvicorn --env-file` 採用）完全版手順書 v1.1

対象: `nova-transcribe/` 直下で開発・起動
目的: ブラウザマイク音声を Bedrock の Nova 2 Sonic 双方向ストリーミングへ送り、**英語の文字起こし（USERロールの textOutput）**をリアルタイム表示・TXT保存

---

## 0. 前提

* **Python 3.12+**（`aws_sdk_bedrock_runtime` が Python>=3.12 前提）
* Bedrock で **Amazon Nova 2 Sonic の Model access を有効化済み**
* 利用リージョン: `ap-northeast-1`（Tokyo 推奨。Nova 2 Sonic 提供リージョンに含まれる）
* 双方向ストリーミングは **ストリーム寿命 8分**（本手順のコードは自動更新を実装）

---

## 1. AWS 側の設定（IAM / Bedrock）

### 1.1 Bedrock: Model access の有効化

1. AWS コンソール → Amazon Bedrock
2. **Model access** で **Nova 2 Sonic** を “Access granted” にする
   （未許可だと Invoke 時に失敗します）

### 1.2 IAM: 開発用ユーザー（例: `bedrock-nova-user`）

すでに作成済みとのことなので、最低限以下を確認します。

* 許可アクション: **`bedrock:InvokeModel`**（双方向ストリーミング呼び出しに必要）

> 参考: 旧情報で `bedrock:InvokeModelWithBidirectionalStream` が出ることがありますが、公式ドキュメント上は `bedrock:InvokeModel` が必須として記載されています。

### 1.3 IAM: アクセスキー作成（ローカル開発用）

1. IAM → ユーザー → `bedrock-nova-user` → **アクセスキーを作成**
2. ユースケース: **ローカルコード**
3. 表示される値を控える（Secret は再表示不可）

   * Access key ID → `AWS_ACCESS_KEY_ID`
   * Secret access key → `AWS_SECRET_ACCESS_KEY`

---

## 2. ローカル開発環境の準備（`nova-transcribe/`）

### 2.1 ディレクトリ作成

```bash
mkdir nova-transcribe
cd nova-transcribe
```

### 2.2 仮想環境（推奨）

**macOS/Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.3 パッケージ導入

```bash
pip install fastapi uvicorn python-dotenv aws_sdk_bedrock_runtime
```

---

## 3. `.env` を作成（Gitに上げない）

### 3.1 `nova-transcribe/.env` を作る

`nova-transcribe` 直下に `.env` を作り、下記を記入します。

```env
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION=ap-northeast-1
```

* IAMユーザーの長期キーの場合、通常 `AWS_SESSION_TOKEN` は不要です（空でOK）

### 3.2 `.gitignore` に追記（必須）

`nova-transcribe/.gitignore` を作成または追記:

```gitignore
.env
.env.*
.venv/
__pycache__/
```

---

## 4. アプリ実装（`main.py`）

`nova-transcribe/main.py` を作成し、以下を貼り付けます。
（英語文字起こしのみ、8分上限回避の自動更新、`.env` は `uvicorn --env-file` で読み込み）

```python
import asyncio
import base64
import json
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

app = FastAPI()

MODEL_ID = "amazon.nova-2-sonic-v1:0"
AWS_REGION_DEFAULT = "ap-northeast-1"

# Bedrock側ストリーム寿命(8分)手前で更新
RENEW_SECONDS = 7 * 60 + 45  # 7m45s


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
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

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

    # .env から region を渡す想定（未設定ならデフォルト）
    import os
    region = os.getenv("AWS_REGION", AWS_REGION_DEFAULT)

    client = get_bedrock_client(region)
    audio_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=256)

    system_prompt = (
        "You are a real-time speech transcription engine.\n"
        "Task: transcribe ONLY what the user says into English text.\n"
        "Rules:\n"
        "- Output ONLY the transcript (no explanations, no extra words).\n"
        "- Do not answer the user, do not add commentary.\n"
        "- Preserve casing and basic punctuation when confident.\n"
    )

    session_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    session: Optional[NovaSonicSession] = None
    output_task: Optional[asyncio.Task] = None

    async def start_session():
        nonlocal session, output_task
        s = NovaSonicSession(client, MODEL_ID)
        await s.open(system_prompt=system_prompt)
        session = s

        if output_task and not output_task.done():
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)

        output_task = asyncio.create_task(read_outputs())

    async def read_outputs():
        nonlocal session
        while not stop_event.is_set() and session and session.is_active:
            msg = await session.recv_once()
            if not msg or "event" not in msg:
                continue

            ev = msg["event"]

            if "contentStart" in ev:
                cs = ev["contentStart"]
                session.current_role = cs.get("role")
                session.current_generation_stage = _extract_generation_stage(cs) or "FINAL"
                continue

            if "textOutput" in ev:
                text = ev["textOutput"].get("content", "")
                if not text:
                    continue

                # USERロールのテキストのみ転写として扱う
                if session.current_role == "USER":
                    stage = (session.current_generation_stage or "FINAL").lower()
                    await websocket.send_text(json.dumps({"type": stage, "text": text}))
                continue

    async def renew_loop():
        while not stop_event.is_set():
            await asyncio.sleep(RENEW_SECONDS)
            if stop_event.is_set():
                return
            async with session_lock:
                if session:
                    await session.close()
                await start_session()
            await websocket.send_text(json.dumps({"type": "info", "text": "Session renewed to avoid the 8-minute limit."}))

    async def recv_from_browser():
        try:
            async for frame in websocket.iter_bytes():
                try:
                    audio_q.put_nowait(frame)
                except asyncio.QueueFull:
                    # 過負荷時はドロップ（メモリ保護）
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()
            try:
                audio_q.put_nowait(None)
            except Exception:
                pass

    async def send_to_bedrock():
        while not stop_event.is_set():
            frame = await audio_q.get()
            if frame is None:
                return
            async with session_lock:
                if session and session.is_active:
                    await session.send_audio(frame)

    async with session_lock:
        await start_session()

    tasks = [
        asyncio.create_task(renew_loop()),
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
    #transcript {
      white-space: pre-wrap;
      min-height: 360px;
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

  <div id="transcript">Press “Start” and speak.</div>
  <div class="hint">
    If you deploy behind HTTPS, this page will automatically use <code>wss://</code>. Microphone access requires a secure context (HTTPS or localhost).
  </div>

<script>
(() => {
  let ws = null;
  let audioContext = null;
  let sourceNode = null;
  let processor = null;

  let finalText = "";
  let partialText = "";

  const TARGET_SR = 16000;
  const FRAME_SAMPLES = 512;
  let pcmBuffer = new Int16Array(0);

  const transcriptDiv = document.getElementById("transcript");
  const statusDiv = document.getElementById("status");
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const downloadBtn = document.getElementById("download");
  const clearBtn = document.getElementById("clear");

  function setStatus(msg) { statusDiv.textContent = msg || ""; }

  function updateView() {
    const merged = finalText + (partialText ? partialText : "");
    transcriptDiv.textContent = merged || "…";
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

    ws.onopen = () => setStatus("Connected. Capturing audio…");
    ws.onclose = () => setStatus("Disconnected.");
    ws.onerror = () => setStatus("WebSocket error.");
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "final") appendFinal(msg.text);
        else if (msg.type === "speculative" || msg.type === "partial") setPartial(msg.text);
        else if (msg.type === "info") setStatus(msg.text);
        else setPartial(msg.text || "");
      } catch {
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

    if (processor) { try { processor.disconnect(); } catch {} }
    if (sourceNode) { try { sourceNode.disconnect(); } catch {} }
    if (audioContext) { try { audioContext.close(); } catch {} }

    processor = null;
    sourceNode = null;
    audioContext = null;

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
```

---

## 5. 起動（`.env` を `--env-file` で読み込む）

`nova-transcribe/` 直下で実行:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

* ブラウザ: `http://localhost:8000`
* Start → マイク許可 → 英語で話す → テキスト表示
* 8分手前で自動更新されると、画面に `Session renewed...` が出ます（継続動作）

---

## 6. 動作確認チェックリスト

1. サーバー起動直後にエラーが出ない（ImportError 等がない）
2. ブラウザで WebSocket が接続され、Status が “Connected…” になる
3. 発話すると transcript が表示される
4. 8分前後で `Session renewed...` が出ても継続する
5. Download TXT が出力できる

---

## 7. よくあるエラーと対処

### AccessDenied / Unauthorized

* IAM に `bedrock:InvokeModel` が付いているか確認
* Bedrock の Model access が許可されているか確認
* `.env` のキーが正しいか（空白や全角混入に注意）

### リージョン不一致

* `.env` の `AWS_REGION` が Nova Sonic 提供リージョンか確認（Tokyo: `ap-northeast-1`）

### マイクが使えない

* `localhost` または HTTPS で開いているか（ブラウザの仕様で制限あり）
* OS 側のマイク権限

---

## 8. セキュリティ運用（最低限）

* `.env` は絶対にコミットしない（`.gitignore` 済みを確認）
* キーは定期ローテーション
* 本番は IAMユーザー長期キーではなく、ロール/SSO/Secrets管理を推奨

---

必要なら、この手順書を「Markdown ファイル（README.md）」としてそのまま保存できる体裁にも整えます（内容は同じで、コピペ用に整形します）。
