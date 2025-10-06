import asyncio
import torch
from pipecat.frames.frames import EndFrame, TextFrame, AudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.vad.silero import SileroVAD
from pipecat.processors.frame_processor import FrameProcessor

class WhisperSTTProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        import whisper
        self.model = whisper.load_model("small")
        
    async def process_frame(self, frame):
        if isinstance(frame, AudioRawFrame):
            result = self.model.transcribe(frame.audio)
            text = result["text"].strip()
            if text and text.lower() not in ["", "you", "thank you"]:
                print(f"👤 User: {text}")
                await self.push_frame(TextFrame(text))
        await self.push_frame(frame)



class OllamaLLMProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        import requests
        self.requests = requests
        
    async def process_frame(self, frame):
        if isinstance(frame, TextFrame):
            response = self.requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama2", "prompt": frame.text, "stream": False}
            )
            reply = response.json()["response"]
            print(f"🤖 Assistant: {reply}")
            await self.push_frame(TextFrame(reply))
        await self.push_frame(frame)


class HuggingFaceTTSProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        from transformers import pipeline
        self.pipe = pipeline(
            "text-to-speech",
            model="microsoft/speecht5_tts",
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
    async def process_frame(self, frame):
        if isinstance(frame, TextFrame):
            speech = self.pipe(frame.text)
            audio_frame = AudioRawFrame(
                audio=speech["audio"].astype(np.float32), 
                sample_rate=speech["sampling_rate"]
            )
            await self.push_frame(audio_frame)
        await self.push_frame(frame)
