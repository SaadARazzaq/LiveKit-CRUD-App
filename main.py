import asyncio
import os
import json
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

load_dotenv()

async def entrypoint(ctx: JobContext):
    # Load personalization settings
    personalization_path = os.getenv("PERSONALIZATION_FILE", "./personalization.json")
    with open(personalization_path, 'r') as f:
        personalization = json.load(f)
    
    system_message = personalization["assistant_instructions"].format(
        ai_assistant_name=personalization["ai_assistant_name"],
        user_name=personalization["user_name"]
    )
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_message,
    )
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(
            model="gpt-4o"
        ),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
        max_nested_fnc_calls=5  # Increased from default 0
    )
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say(f"Hi {personalization['user_name']}, I'm {personalization['ai_assistant_name']}. How can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))