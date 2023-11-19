import asyncio
import livekit.rtc as rtc
from livekit import agents
from livekit.plugins import core
from livekit.plugins.vad import VADPlugin
from livekit.plugins.openai import WhisperLocalTranscriber
import logging
from typing import AsyncIterator


async def stt_agent(ctx: agents.JobContext):
    logging.info("starting stt agent")
    # agent is connected to the room as a participant

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(ctx.room, track, participant))


async def process_track(room: rtc.Room, track: rtc.Track, participant: rtc.RemoteParticipant):
    audio_stream = rtc.AudioStream(track)
    input_iterator = core.PluginIterator.create(audio_stream)
    vad_plugin = VADPlugin(
        left_padding_ms=250, silence_threshold_ms=500)
    stt_plugin = WhisperLocalTranscriber()

    await vad_plugin \
        .set_input(input_iterator) \
        .filter(lambda data: data.type == core.VADPluginResultType.FINISHED) \
        .pipe(stt_plugin) \
        .map_async(lambda text_stream, metadata: process_stt(room, participant, text_stream, metadata)) \
        .run()


async def process_stt(room: rtc.Room,
                      participant: rtc.RemoteParticipant,
                      text_stream: AsyncIterator[str],
                      metadata: core.PluginIterator.ResultMetadata,
                      ):
    complete_text = ""
    async for stt_r in text_stream:
        complete_text += stt_r.text
    logging.info(f"{participant.identity}: {complete_text}")
    asyncio.create_task(room.local_participant.publish_data(complete_text))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        await job_request.accept(
            stt_agent,
            should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO,
        )

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    agents.run_app(worker)
