import asyncio
import gzip
import json
import os
import struct
import uuid
import websockets
from livekit import rtc
from livekit.agents import stt

from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

# 协议常量
PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

# 消息类型
FULL_CLIENT_REQUEST = 0b0001
AUDIO_ONLY_REQUEST = 0b0010
FULL_SERVER_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# 消息类型特定标志
NO_SEQUENCE = 0b0000
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_WITH_SEQUENCE = 0b0011

# 序列化方法
NO_SERIALIZATION = 0b0000
JSON = 0b0001

# 压缩类型
NO_COMPRESSION = 0b0000
GZIP = 0b0001

# WebSocket URL
BASE_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"

# 日志记录
import logging
logger = logging.getLogger("volcengine-stt")

class STT(stt.STT):
    def __init__(
        self,
        app_id: str,
        access_token: str,
        language: str = "zh-CN",
        sample_rate: int = 16000,
        resource_id: str = "volc.bigasr.sauc.duration",
        base_url: str = BASE_URL,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True),
            language=language,
            sample_rate=sample_rate,
        )
        self.app_id = app_id or os.environ.get("VOLCENGINE_API_APP_KEY")
        self.access_token = access_token or os.environ.get("VOLCENGINE_API_ACCESS_KEY")
        self.resource_id = resource_id
        self.base_url = base_url

    def stream(self):
        return SpeechStream(self)

class SpeechStream(stt.SpeechStream):
    def __init__(self, stt: STT):
        super().__init__(stt)
        self.stt = stt
        self.seq = 1
        self.speaking = False
        self._ws = None

    async def __aenter__(self):
        extra_headers = {
            "X-Api-App-Key": self.stt.app_id,
            "X-Api-Access-Key": self.stt.access_token,
            "X-Api-Resource-Id": self.stt.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self._ws = await websockets.connect(self.stt.base_url, extra_headers=extra_headers)
        await self._send_full_client_request()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._ws:
            await self._ws.close()

    async def _send_full_client_request(self):
        request_params = {
            "user": {"uid": "example_uid"},
            "audio": {
                "format": "pcm",
                "sample_rate": self.stt.sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_punc": True,
            }
        }
        payload_bytes = gzip.compress(json.dumps(request_params).encode("utf-8"))
        full_client_request = bytearray(self._generate_header(message_type_specific_flags=POS_SEQUENCE))
        full_client_request.extend(self._generate_before_payload(self.seq))
        full_client_request.extend(len(payload_bytes).to_bytes(4, 'big'))
        full_client_request.extend(payload_bytes)
        await self._ws.send(full_client_request)
        self.seq += 1

    async def push_audio(self, frame: rtc.AudioFrame):
        chunk = frame.data.tobytes()
        await self._send_audio_chunk(chunk, False)

    async def flush(self):
        await self._send_audio_chunk(b"", True)

    async def _send_audio_chunk(self, chunk: bytes, is_last: bool):
        payload_bytes = gzip.compress(chunk)
        flags = NEG_WITH_SEQUENCE if is_last else POS_SEQUENCE
        audio_only_request = bytearray(self._generate_header(message_type=AUDIO_ONLY_REQUEST, message_type_specific_flags=flags))
        seq = -self.seq if is_last else self.seq
        audio_only_request.extend(self._generate_before_payload(seq))
        audio_only_request.extend(len(payload_bytes).to_bytes(4, 'big'))
        audio_only_request.extend(payload_bytes)
        await self._ws.send(audio_only_request)
        if not is_last:
            self.seq += 1

    async def run(self):
        while True:
            res = await self._ws.recv()
            result = self._parse_response(res)
            if "payload_msg" in result:
                text = result["payload_msg"].get("result", {}).get("text", "")
                is_final = result.get("is_last_package", False)
                if text:
                    if not self.speaking:
                        self.speaking = True
                        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
                    event_type = stt.SpeechEventType.FINAL_TRANSCRIPT if is_final else stt.SpeechEventType.INTERIM_TRANSCRIPT
                    event = stt.SpeechEvent(
                        type=event_type,
                        request_id="",
                        alternatives=[
                            stt.SpeechData(
                                language=self.stt.language,
                                start_time=0,
                                end_time=0,
                                confidence=0.9,
                                text=text,
                            )
                        ],
                    )
                    self._event_ch.send_nowait(event)
                if is_final and self.speaking:
                    self.speaking = False
                    self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _generate_header(
        self,
        message_type=FULL_CLIENT_REQUEST,
        message_type_specific_flags=NO_SEQUENCE,
        serial_method=JSON,
        compression_type=GZIP,
        reserved_data=0x00
    ):
        header = bytearray()
        header_size = 1
        header.append((PROTOCOL_VERSION << 4) | header_size)
        header.append((message_type << 4) | message_type_specific_flags)
        header.append((serial_method << 4) | compression_type)
        header.append(reserved_data)
        return header

    def _generate_before_payload(self, sequence: int):
        before_payload = bytearray()
        before_payload.extend(sequence.to_bytes(4, 'big', signed=True))
        return before_payload

    def _parse_response(self, res):
        protocol_version = res[0] >> 4
        header_size = res[0] & 0x0f
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0f
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0f
        reserved = res[3]
        header_extensions = res[4:header_size * 4]
        payload = res[header_size * 4:]
        result = {
            'is_last_package': False,
        }
        if message_type_specific_flags & 0x01:
            seq = int.from_bytes(payload[:4], "big", signed=True)
            result['payload_sequence'] = seq
            payload = payload[4:]
        if message_type_specific_flags & 0x02:
            result['is_last_package'] = True
        if message_type == FULL_SERVER_RESPONSE:
            payload_size = int.from_bytes(payload[:4], "big", signed=True)
            payload_msg = payload[4:]
        elif message_type == SERVER_ACK:
            seq = int.from_bytes(payload[:4], "big", signed=True)
            result['seq'] = seq
            if len(payload) >= 8:
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                payload_msg = payload[8:]
        elif message_type == SERVER_ERROR_RESPONSE:
            code = int.from_bytes(payload[:4], "big", signed=False)
            result['code'] = code
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
        if message_compression == GZIP:
            payload_msg = gzip.decompress(payload_msg)
        if serialization_method == JSON:
            payload_msg = json.loads(str(payload_msg, "utf-8"))
        elif serialization_method != NO_SERIALIZATION:
            payload_msg = str(payload_msg, "utf-8")
        result['payload_msg'] = payload_msg
        result['payload_size'] = payload_size
        return result