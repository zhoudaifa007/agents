# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass

import aiohttp
import websockets

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ._utils import _to_volcengine_url
from .log import logger
from .models import VolcengineVoiceTypes

BASE_URL = "wss://openspeech.bytedance.com/api/v1/tts"
NUM_CHANNELS = 1

@dataclass
class _TTSOptions:
    voice_type: VolcengineVoiceTypes | str
    encoding: str
    sample_rate: int
    speed: float
    volume: float
    pitch: float
    word_tokenizer: tokenize.WordTokenizer

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_type: VolcengineVoiceTypes | str = "BV001_STREAMING",
        encoding: str = "pcm",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 1.0,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        access_token: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = BASE_URL,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Volcengine TTS.

        Args:
            voice_type: Voice type to use. Defaults to "BV001_STREAMING".
            encoding: Audio encoding to use. Defaults to "pcm".
            sample_rate: Sample rate of audio. Defaults to 16000.
            speed: Speech speed. Defaults to 1.0.
            volume: Speech volume. Defaults to 1.0.
            pitch: Speech pitch. Defaults to 1.0.
            app_id: Your Volcengine app ID. If not provided, will look for VOLCENGINE_APP_ID environment variable.
            access_token: Your Volcengine access token. If not provided, will look for VOLCENGINE_ACCESS_TOKEN environment variable.
            base_url: Base URL for Volcengine TTS API. Defaults to "wss://openspeech.bytedance.com/api/v1/tts".
            word_tokenizer: Tokenizer for processing text. Defaults to basic WordTokenizer.
            http_session: Optional aiohttp session to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._app_id = app_id if is_given(app_id) else os.environ.get("VOLCENGINE_APP_ID")
        self._access_token = access_token if is_given(access_token) else os.environ.get("VOLCENGINE_ACCESS_TOKEN")
        if not self._app_id or not self._access_token:
            raise ValueError("Volcengine app_id and access_token are required")

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            voice_type=voice_type,
            encoding=encoding,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            pitch=pitch,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session
        self._base_url = base_url
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[websockets.WebSocketClientProtocol](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self) -> websockets.WebSocketClientProtocol:
        config = {
            "app_id": self._app_id,
            "access_token": self._access_token,
            "voice_type": self._opts.voice_type,
            "audio_format": self._opts.encoding,
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "volume": self._opts.volume,
            "pitch": self._opts.pitch,
        }
        ws = await websockets.connect(
            _to_volcengine_url(config, self._base_url, websocket=True),
            ping_interval=None,
        )
        return ws

    async def _close_ws(self, ws: websockets.WebSocketClientProtocol):
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_type: NotGivenOr[VolcengineVoiceTypes | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        pitch: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_type):
            self._opts.voice_type = voice_type
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pitch):
            self._opts.pitch = pitch

        for stream in self._streams:
            stream.update_options(
                voice_type=voice_type,
                sample_rate=sample_rate,
                speed=speed,
                volume=volume,
                pitch=pitch,
            )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            base_url=self._base_url,
            app_id=self._app_id,
            access_token=self._access_token,
            conn_options=conn_options,
            opts=self._opts,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            base_url=self._base_url,
            app_id=self._app_id,
            access_token=self._access_token,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        base_url: str,
        app_id: str,
        access_token: str,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._base_url = base_url
        self._app_id = app_id
        self._access_token = access_token

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        try:
            async with websockets.connect(
                _to_volcengine_url({}, self._base_url, websocket=True)
            ) as ws:
                config = {
                    "app_id": self._app_id,
                    "access_token": self._access_token,
                    "text": self._input_text,
                    "voice_type": self._opts.voice_type,
                    "audio_format": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                    "speed": self._opts.speed,
                    "volume": self._opts.volume,
                    "pitch": self._opts.pitch,
                }
                await ws.send(json.dumps(config))

                while True:
                    data = await ws.recv()
                    if isinstance(data, str):
                        continue
                    for frame in audio_bstream.write(data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                            )
                        )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        base_url: str,
        app_id: str,
        access_token: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._base_url = base_url
        self._app_id = app_id
        self._access_token = access_token
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        voice_type: NotGivenOr[VolcengineVoiceTypes | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        pitch: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_type):
            self._opts.voice_type = voice_type
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pitch):
            self._opts.pitch = pitch
        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None
            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _run_segments(ws: websockets.WebSocketClientProtocol):
            nonlocal closing_ws
            async for word_stream in self._segments_ch:
                async for word in word_stream:
                    config = {
                        "app_id": self._app_id,
                        "access_token": self._access_token,
                        "text": f"{word.token} ",
                        "voice_type": self._opts.voice_type,
                        "audio_format": self._opts.encoding,
                        "sample_rate": self._opts.sample_rate,
                        "speed": self._opts.speed,
                        "volume": self._opts.volume,
                        "pitch": self._opts.pitch,
                    }
                    self._mark_started()
                    await ws.send(json.dumps(config))

            closing_ws = True

        async def recv_task(ws: websockets.WebSocketClientProtocol):
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                segment_id=segment_id,
            )

            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    resp = json.loads(msg)
                    if resp.get("status") == "finished":
                        for frame in audio_bstream.flush():
                            emitter.push(frame)
                        emitter.flush()
                        break
                    continue

                for frame in audio_bstream.write(msg):
                    emitter.push(frame)

        ws: websockets.WebSocketClientProtocol | None = None
        while True:
            try:
                ws = await self.tts._pool.get()
                closing_ws = False

                tasks = [
                    asyncio.create_task(_tokenize_input()),
                    asyncio.create_task(_run_segments(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if wait_reconnect_task not in done:
                        break
                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)

            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                if ws is not None:
                    self.tts._pool.release(ws)