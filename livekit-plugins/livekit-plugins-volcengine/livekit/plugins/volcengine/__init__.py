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

"""Volcengine plugin for LiveKit Agents

Support for speech-to-text and text-to-speech with Volcengine (https://www.volcengine.com).

See Volcengine TTS/STT documentation for more information:
- TTS: https://www.volcengine.com/docs/6561/125674
- STT: https://www.volcengine.com/docs/6561/61496
"""

from .stt import STT, SpeechStream
from .tts import TTS
from .version import __version__

__all__ = ["STT", "SpeechStream", "__version__", "TTS"]

from livekit.agents import Plugin
from .log import logger

class VolcenginePlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

Plugin.register_plugin(VolcenginePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False