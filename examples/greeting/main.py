import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from pydub import AudioSegment
import logging

# 降低 pydub 日志级别
logging.getLogger("pydub").setLevel(logging.INFO)

# 加载环境变量
load_dotenv()
# 配置LiveKit服务器信息
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://your-livekit-server")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "your_api_key")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "your_api_secret")
# 欢迎语音频文件路径（相对于项目根目录）
GREETING_FILE_PATH = Path(__file__).parent / "HeartofCourage.mp3"
print(f"欢迎语音频文件路径：{GREETING_FILE_PATH}")


async def publish_audio_to_room(room: rtc.Room, file_path: Path):
    """将欢迎语音频发布到LiveKit房间"""
    audio_track = None
    audio_source = None
    try:
        print(f"开始加载音频文件: {file_path}")
        audio = AudioSegment.from_mp3(file_path)[:10000]  # 裁剪前10秒
        audio = audio - 5  # 降低5dB
        audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)

        # 创建 AudioSource
        audio_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
        # 使用 AudioSource 创建 LocalAudioTrack
        audio_track = rtc.LocalAudioTrack.create_audio_track(
            name="greeting_audio",
            source=audio_source
        )
        await room.local_participant.publish_track(audio_track)
        print(f"音频轨道已发布: {file_path}")

        # 将音频数据分块发送（每20ms一帧）
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        frame_duration = 0.02  # 20ms帧
        samples_per_frame = int(48000 * frame_duration)
        for i in range(0, len(samples), samples_per_frame):
            frame_samples = samples[i:i + samples_per_frame]
            if len(frame_samples) < samples_per_frame:
                frame_samples = np.pad(frame_samples, (0, samples_per_frame - len(frame_samples)))
            audio_frame = rtc.AudioFrame(
                data=frame_samples.tobytes(),
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=len(frame_samples)
            )
            await audio_source.capture_frame(audio_frame)
            await asyncio.sleep(frame_duration)
        print(f"欢迎语音频 {file_path} 播放完成")
    except Exception as e:
        print(f"发布欢迎语音频失败: {e}")
    finally:
        if audio_track:
            await room.local_participant.unpublish_track(audio_track)
            print("音频轨道已取消发布")
        if audio_source:
            audio_source.close()
            print("音频源已关闭")


async def entrypoint(ctx: JobContext):
    try:
        # 连接到LiveKit房间
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        print(f"Welcome Agent已连接到房间: {ctx.room.name}")
        # 获取当前房间中的远程参与者
        participants = ctx.room.remote_participants
        print(f"房间中的参与者：{[p.identity for p in participants.values()]}")

        # 检查音频文件是否存在
        if not GREETING_FILE_PATH.exists():
            print(f"错误：音频文件 {GREETING_FILE_PATH} 不存在")
            return

        # 为现有参与者播放欢迎音频（异步执行）
        for participant in participants.values():
            print(f"发现现有用户 {participant.identity}")
            asyncio.create_task(publish_audio_to_room(ctx.room, GREETING_FILE_PATH))

        @ctx.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            print(f"用户 {participant.identity} 进入了房间")
            if GREETING_FILE_PATH.exists():
                asyncio.create_task(publish_audio_to_room(ctx.room, GREETING_FILE_PATH))
            else:
                print(f"欢迎语音频文件 {GREETING_FILE_PATH} 不存在")

        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            print(f"用户 {participant.identity} 离开了房间")

        # 保持Agent运行
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Welcome Agent运行出错: {e}")


def run():
    """运行LiveKit Welcome Agent"""
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        ws_url=LIVEKIT_URL
    )
    cli.run_app(worker_options)


if __name__ == "__main__":
    run()