import logging

import os
from dotenv import load_dotenv

load_dotenv('.env.local')  # Ensure the correct path to your .env.local file
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import google, openai, silero, turn_detector, deepgram, elevenlabs

load_dotenv()
logger = logging.getLogger("voice-assistant")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# An example Voice Agent using Google STT, Gemini 2.0 Flash, and Google TTS.
# Prerequisites:
# 1. livekit-plugins-openai[vertex] package installed
# 2. save your service account credentials and set the following environments:
#    * GOOGLE_APPLICATION_CREDENTIALS to the path of the service account key file
#    * GOOGLE_CLOUD_PROJECT to your Google Cloud project ID
# 3. the following services are enabled on your Google Cloud project:
#    * Vertex AI
#    * Cloud Speech-to-Text API
#    * Cloud Text-to-Speech API

# Read more about authentication with Google: https://cloud.google.com/docs/authentication/application-default-credentials


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    voice_config = elevenlabs.Voice(
        id="cgSgspJ2msm6clMCkdW9",
        name="Jessica",
        category="premade",
        settings=elevenlabs.VoiceSettings(
            stability=0.71,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    elevenlabs_tts = elevenlabs.TTS(
        voice=voice_config,
        model="eleven_flash_v2_5",
        api_key=os.getenv("ELEVEN_API_KEY"),
        encoding="mp3_22050_32",
        streaming_latency=3,
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
    )

    # Log the voice details using the `voice_config` object
    logger.info(f"Using ElevenLabs TTS with voice ID: {voice_config.id} and name: {voice_config.name}")

    # Create and start the voice agent
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM.with_vertex(model="google/gemini-2.0-flash-exp"),
        tts=elevenlabs_tts,
        chat_ctx=initial_ctx,
        turn_detector=turn_detector.EOUModel(),
    )

    agent.start(ctx.room, participant)

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    await agent.say(
        "Hi there, this is Gemini, how can I help you today?", allow_interruptions=False
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )