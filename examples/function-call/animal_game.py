from __future__ import annotations

import asyncio
import logging
from typing import Annotated
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    multimodal,
)
from livekit.plugins import openai

load_dotenv()

logger = logging.getLogger("animal-image-worker")
logger.setLevel(logging.INFO)

# 本地图片目录
IMAGE_DIR = Path("/path/to/images")

# 本地图片映射
ANIMAL_IMAGE_MAP = {
    "dog": IMAGE_DIR / "dog.jpg",
    "cat": IMAGE_DIR / "cat.jpg",
    "bird": IMAGE_DIR / "bird.jpg",
    "elephant": IMAGE_DIR / "elephant.jpg",
    "lion": IMAGE_DIR / "lion.jpg",
}


async def entrypoint(ctx: JobContext):
    logger.info("Starting the animal image worker...")

    # FunctionContext 用于定义支持的 LLM 函数
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def show_animal_image(
        animal: Annotated[str, llm.TypeInfo(description="Name of the animal to show the image for")]
    ):
        """Display an image for the specified animal."""
        logger.info(f"Received animal: {animal}")
        animal_key = animal.lower()

        if animal_key in ANIMAL_IMAGE_MAP:
            image_path = ANIMAL_IMAGE_MAP[animal_key]
            if image_path.exists():
                # 如果需要返回 HTTP URL，可启动一个图片服务（如 Flask 或 FastAPI）并生成图片 URL
                logger.info(f"Displaying image for {animal}: {image_path}")
                return f"Here is an image of a {animal}: {image_path}"
            else:
                logger.error(f"Image file not found for {animal}: {image_path}")
                return f"Sorry, the image file for {animal} is missing."
        else:
            logger.warning(f"No image found for {animal}")
            return f"Sorry, I don't have an image for {animal}."

    # 自动订阅音频流
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    # 设置实时语音识别和响应
    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions="You are an assistant that shows animal images upon request.",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            ),
        ),
        fnc_ctx=fnc_ctx,
    )

    # 启动代理并监听房间和参与者
    agent.start(ctx.room, participant)

    @agent.on("agent_speech_committed")
    async def on_speech_committed(msg: llm.ChatMessage):
        """处理语音识别完成事件并显示图片"""
        logger.info(f"User said: {msg.text}")
        animal_key = msg.text.lower()

        if animal_key in ANIMAL_IMAGE_MAP:
            image_path = ANIMAL_IMAGE_MAP[animal_key]
            if image_path.exists():
                logger.info(f"Found image for {msg.text}: {image_path}")
                await agent.say(f"Here is an image of a {msg.text}: {image_path}")
            else:
                logger.error(f"Image file not found for {msg.text}: {image_path}")
                await agent.say(f"Sorry, the image file for {msg.text} is missing.")
        else:
            logger.warning(f"No image available for: {msg.text}")
            await agent.say(f"Sorry, I don't have an image for {msg.text}.")

    logger.info("Agent is ready to receive audio input...")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))