import asyncio
import copy
import logging
from collections import deque
from typing import Annotated, List

from livekit import agents, rtc
from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatRole,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit.plugins.elevenlabs import Voice

MAIN_PROMPT = "You are a funny bot created by LiveKit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
RUNDOWN_PROMPT = """You are an assistant acting as the founder and CEO of the company, LiveKit. You will answer any questions users have when they call in to speak to you
LiveKit rec""ently raised a $22.5M round of funding to build infrastructure for realtime voice and video-driven AI applications
Users wil""l call in and speak with you using a telephone. You will converse with them using your voice
In gen""eral, you should use short and concise responses and avoid using unpronounceable punctuation.
You should also act professionally, but it's OK to speak casually or informally, and occasionally tell a self-deprecating joke or two."""
MAIN_GREETING = "Hey, how can I help you today?"
RUNDOWN_GREETING = "Hey hey, this is LiveKit's founder and C3PO, what's on your mind?"


async def entrypoint(ctx: JobContext):
    rundown = ctx.room.name.startswith("rundown")
    prompt = RUNDOWN_PROMPT if rundown else MAIN_PROMPT
    greeting = RUNDOWN_GREETING if rundown else MAIN_GREETING
    initial_ctx = ChatContext(messages=[ChatMessage(role=ChatRole.SYSTEM, text=prompt)])

    gpt = openai.LLM(
        model="gpt-4o",
    )
    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs.TTS(
            encoding="pcm_44100",
            voice=Voice(id="sR1Nne6UFqWWc3gpP4Ja", name="Russ", category="standard"),
        ),
        chat_ctx=initial_ctx,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer_from_text(text: str):
        chat_ctx = copy.deepcopy(assistant.chat_context)
        chat_ctx.messages.append(ChatMessage(role=ChatRole.USER, text=text))

        stream = await gpt.chat(chat_ctx)
        await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if not msg.message:
            return

        asyncio.create_task(_answer_from_text(msg.message))

    assistant.start(ctx.room)

    await asyncio.sleep(0.5)
    await assistant.say(greeting, allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
