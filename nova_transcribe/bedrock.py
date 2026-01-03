import base64
import json
import logging
import uuid
from typing import Optional

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithResponseStreamInput,
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_core.interceptors import Interceptor
from smithy_core.interfaces import TypedProperties
from smithy_http import Field


logger = logging.getLogger("main")


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

        await self._send_event(
            json.dumps(
                {
                    "event": {
                        "sessionStart": {
                            "inferenceConfiguration": {
                                "maxTokens": 1024,
                                "topP": 0.9,
                                "temperature": 0.0,
                            },
                            "turnDetectionConfiguration": {
                                "endpointingSensitivity": "HIGH"
                            },
                        }
                    }
                }
            )
        )

        await self._send_event(
            json.dumps(
                {
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
                }
            )
        )

        # SYSTEM text
        await self._send_event(
            json.dumps(
                {
                    "event": {
                        "contentStart": {
                            "promptName": self.prompt_name,
                            "contentName": self.system_content_name,
                            "type": "TEXT",
                            "interactive": False,
                            "role": "SYSTEM",
                            "textInputConfiguration": {
                                "mediaType": "text/plain"
                            },
                        }
                    }
                }
            )
        )
        await self._send_event(
            json.dumps(
                {
                    "event": {
                        "textInput": {
                            "promptName": self.prompt_name,
                            "contentName": self.system_content_name,
                            "content": system_prompt,
                        }
                    }
                }
            )
        )
        await self._send_event(
            json.dumps(
                {
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": self.system_content_name,
                        }
                    }
                }
            )
        )

        # USER audio container
        await self._send_event(
            json.dumps(
                {
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
                }
            )
        )
        logger.info("✓ Bedrock session initialized and ready")

    async def send_audio(self, pcm16le_bytes: bytes) -> None:
        if not self.is_active:
            return
        b64 = base64.b64encode(pcm16le_bytes).decode("utf-8")
        await self._send_event(
            json.dumps(
                {
                    "event": {
                        "audioInput": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name,
                            "content": b64,
                        }
                    }
                }
            )
        )

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
            await self._send_event(
                json.dumps(
                    {
                        "event": {
                            "contentEnd": {
                                "promptName": self.prompt_name,
                                "contentName": self.audio_content_name,
                            }
                        }
                    }
                )
            )
            await self._send_event(
                json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}})
            )
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
    max_tokens: int,
) -> object:
    """
    Claude をストリーミング呼び出しし、テキスト delta を yield する。
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
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

