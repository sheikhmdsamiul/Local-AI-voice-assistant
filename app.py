import asyncio
import numpy as np


from pipecat.frames.frames import TextFrame, FrameProcessor, AudioRawFrame

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

