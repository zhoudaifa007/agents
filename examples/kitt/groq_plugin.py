# Copyright 2023 LiveKit, Inc.
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

import os
import httpx
from groq import AsyncGroq, APITimeoutError
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional
from enum import Enum

GroqMessageRole = Enum("MessageRole", ["system", "user", "assistant", "function"])


class GroqModels(Enum):
    Mixtral7B = "mixtral-8x7b-32768"
    Gemma7B = "gemma-7b-it"
    Llama2_70B = "llama2-70b-4096"


@dataclass
class GroqMessage:
    role: GroqMessageRole
    content: str

    def to_api(self):
        return {"role": self.role.name, "content": self.content}


class GroqPlugin:
    """Groq Plugin"""

    def __init__(self, prompt: str, message_capacity: int, model: str):
        """
        Args:
            prompt (str): First 'system' message sent to the chat that prompts the assistant
            message_capacity (int): Maximum number of messages to send to the chat
            model (str): Which model to use (i.e. 'gpt-3.5-turbo')
        """
        self._model = model
        self._client = AsyncGroq(
            api_key=os.environ["GROQ_API_KEY"],
            timeout=httpx.Timeout(10.0, read=5.0, connect=10.0),
        )
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: List[GroqMessage] = []
        self._producing_response = False
        self._needs_interrupt = False

    def interrupt(self):
        """Interrupt a currently streaming response (if there is one)"""
        if self._producing_response:
            self._needs_interrupt = True

    async def aclose(self):
        pass

    async def send_system_prompt(self) -> AsyncIterable[str]:
        """Send the system prompt to the chat and generate a streamed response

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """
        async for text in self.add_message(None):
            yield text

    async def add_message(self, message: Optional[GroqMessage]) -> AsyncIterable[str]:
        """Add a message to the chat and generate a streamed response

        Args:
            message (GroqMessage): The message to add

        Returns:
            AsyncIterable[str]: Streamed Groq response
        """

        if message is not None:
            self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)

        async for text in self._generate_text_streamed(self._model):
            yield text

    async def _generate_text_streamed(self, model: str) -> AsyncIterable[str]:
        prompt_message = GroqMessage(role=GroqMessageRole.system, content=self._prompt)
        try:
            chat_messages = [m.to_api() for m in self._messages]
            chat_stream = await self._client.chat.completions.create(
                model=model,
                stream=True,
                temperature=0.5,
                max_tokens=1024,
                stop=None,
                messages=[prompt_message.to_api()] + chat_messages,
            )
        except APITimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        complete_response = ""

        async for chunk in chat_stream:
            if chunk is None:
                break
            content = chunk.choices[0].delta.content
            if content is not None:
                complete_response += content
                yield content

        self._messages.append(
            GroqMessage(role=GroqMessageRole.assistant, content=complete_response)
        )
        self._producing_response = False
