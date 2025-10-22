import asyncio
import numpy as np
import torch
import whisper
import logging
import os
from typing import Optional

from pipecat.frames.frames import EndFrame, TextFrame, AudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.processors.frame_processor import FrameProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperSTTProcessor(FrameProcessor):
    def __init__(self, model_size: str = "base", window_sec: float = 1.0, overlap_sec: float = 0.5):
        super().__init__()
        logger.info(f"Loading Whisper model: {model_size}")
        try:
            self.model = whisper.load_model(model_size)
            self.sample_rate = 16000
            self.window_size = int(self.sample_rate * window_sec)
            self.overlap_size = int(self.sample_rate * overlap_sec)
            self.audio_buffer = np.array([], dtype=np.float32)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.exception("Failed to load Whisper model")
            raise

    async def process_frame(self, frame):
        if isinstance(frame, AudioRawFrame):
            try:
                # Validate incoming format when available
                if hasattr(frame, 'sample_rate') and frame.sample_rate != self.sample_rate:
                    # Drop or resample; for now drop with warning
                    logger.warning(f"Unexpected sample rate {getattr(frame, 'sample_rate', None)}; expected {self.sample_rate}. Dropping frame.")
                    return
                # Convert audio to numpy array (mono int16 -> float32 [-1,1])
                audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                # Append to buffer
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_data)) if self.audio_buffer.size else audio_data
                # Process in fixed windows
                while self.audio_buffer.size >= self.window_size:
                    chunk = self.audio_buffer[:self.window_size]
                    # Transcribe
                    result = self.model.transcribe(
                        chunk,
                        fp16=torch.cuda.is_available(),
                        no_speech_threshold=0.6
                    )
                    text = (result.get("text") or "").strip()
                    skip_texts = {"", "you", "thank you", "thanks", "hello", "hi"}
                    if text and len(text) > 2 and text.lower() not in skip_texts and result.get("no_speech_prob", 0) < 0.8:
                        logger.info(f"🎤 User: {text}")
                        await self.push_frame(TextFrame(text))
                    # Keep overlap
                    start = self.window_size - self.overlap_size
                    self.audio_buffer = self.audio_buffer[start:]
            except Exception:
                logger.exception("STT processing error")
                self.audio_buffer = np.array([], dtype=np.float32)
        else:
            await self.push_frame(frame)

class OllamaLLMProcessor(FrameProcessor):
    def __init__(self, model: str = "llama2"):
        super().__init__()
        import requests
        self.requests = requests
        self.model = model
        self.conversation_history = []
        self.max_history = 3
        # Avoid newline stop which truncates responses
        self.stop_tokens = ["User:", "Assistant:", "###"]

    async def process_frame(self, frame):
        if isinstance(frame, TextFrame):
            user_text = frame.text.strip()
            
            if len(user_text) < 3:
                return
                
            try:
                # Add user message to history
                self.conversation_history.append(f"User: {user_text}")
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-(self.max_history * 2):]
                
                # Build context from history
                context = "\n".join(self.conversation_history[-self.max_history * 2:])
                
                prompt = f"""You are a concise, helpful assistant. Continue the conversation in 1-3 sentences.

{context}
Assistant:"""
                
                logger.info("Calling Ollama for response...")
                
                response = self.requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 120,
                            "stop": self.stop_tokens
                        }
                    },
                    timeout=45
                )
                
                if response.status_code == 200:
                    data = response.json()
                    reply = (data.get("response") or "").strip()
                    # Do not depend on "Assistant:" prefix in reply; just use text
                    if reply:
                        logger.info(f"🤖 Assistant: {reply}")
                        self.conversation_history.append(f"Assistant: {reply}")
                        await self.push_frame(TextFrame(reply))
                    else:
                        logger.warning("Received empty response from Ollama")
                        await self.push_frame(TextFrame("I didn't understand that. Could you please repeat?"))
                else:
                    safe_text = response.text if isinstance(response.text, str) else ""
                    logger.error(f"Ollama API error {response.status_code}: {safe_text}")
                    await self.push_frame(TextFrame("I'm having trouble processing your request. Please try again."))
                    
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                await self.push_frame(TextFrame("Sorry, I encountered an error. Please try again."))
        else:
            await self.push_frame(frame)

class HuggingFaceTTSProcessor(FrameProcessor):
    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        super().__init__()
        self.model_name = model_name
        self.pipe = None
        self.setup_tts()

    def setup_tts(self):
        """Initialize TTS pipeline using a simple, widely supported model.
        Note: transformers text-to-speech pipelines vary; for robustness, consider alternative libraries.
        """
        try:
            from transformers import pipeline
            use_cuda = torch.cuda.is_available()
            device = 0 if use_cuda else -1
            logger.info(f"Loading TTS model: {self.model_name} on {'GPU' if use_cuda else 'CPU'}")
            # Try a basic pipeline; fall back if not supported
            self.pipe = pipeline(
                task="text-to-speech",
                model=self.model_name,
                device=device
            )
            logger.info("TTS model loaded successfully")
        except Exception:
            logger.exception(f"Failed to load TTS model {self.model_name}")
            self.try_fallback_tts()

    def try_fallback_tts(self):
        """Try to load a fallback TTS model"""
        fallback_models = [
            # Use models that commonly work with the pipeline, adjust if needed
            "espnet/kan-bayashi_ljspeech_vits"
        ]
        for model in fallback_models:
            try:
                from transformers import pipeline
                logger.info(f"Trying fallback TTS model: {model}")
                self.pipe = pipeline(task="text-to-speech", model=model)
                self.model_name = model
                logger.info(f"Fallback TTS model {model} loaded successfully")
                return
            except Exception:
                logger.exception(f"Failed to load fallback {model}")
                continue
        logger.error("All TTS models failed to load")
        self.pipe = None

    async def process_frame(self, frame):
        if isinstance(frame, TextFrame):
            if not self.pipe:
                logger.error("TTS not available - no model loaded")
                return
            text = frame.text.strip()
            if len(text) < 2:
                return
            try:
                logger.info("Generating speech from text...")
                result = self.pipe(text)
                # Result structure can vary; support common formats
                waveform = None
                sample_rate = 16000
                if isinstance(result, dict):
                    if "audio" in result:
                        waveform = result["audio"]
                    if "sampling_rate" in result:
                        sample_rate = int(result["sampling_rate"]) or sample_rate
                if waveform is None:
                    # Some pipelines return list of dicts
                    if isinstance(result, list) and result and isinstance(result[0], dict) and "audio" in result[0]:
                        waveform = result[0]["audio"]
                        if "sampling_rate" in result[0]:
                            sample_rate = int(result[0]["sampling_rate"]) or sample_rate
                if waveform is None:
                    logger.error("Unsupported TTS output format")
                    return
                # Ensure numpy array float32 mono
                waveform = np.array(waveform, dtype=np.float32)
                if waveform.ndim > 1:
                    # Downmix to mono
                    waveform = waveform.mean(axis=0)
                # Convert to int16 bytes
                audio_data = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                audio_frame = AudioRawFrame(audio=audio_data, sample_rate=sample_rate, channels=1)
                await self.push_frame(audio_frame)
                logger.info("Speech generated and sent successfully")
            except Exception:
                logger.exception("TTS generation failed")
        else:
            await self.push_frame(frame)

async def main():
    logger.info("Starting AI Voice Assistant with Local Audio Transport")
    
    # Check essential dependencies
    try:
        import requests
        logger.info("✓ Required dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.info("Please install: pip install requests transformers torch openai-whisper daily-pipe")
        return

    # Verify Ollama is running
    try:
        import requests
        test_response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if test_response.status_code == 200:
            logger.info("Ollama is running and accessible")
        else:
            logger.error("Ollama responded with error. Please check if it's running properly.")
            return
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        logger.info("Please make sure Ollama is running: https://ollama.com")
        return

    # Using local audio transport; no API keys required.

    # Initialize local audio transport (microphone -> pipeline -> speakers)
    try:
        transport = LocalAudioTransport(
            sample_rate=16000,
            channels=1,
            blocksize=1024
        )
        logger.info("Local Audio Transport initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Local Audio Transport: {e}")
        return

    # Create processors
    try:
        stt = WhisperSTTProcessor(model_size="base")
        llm = OllamaLLMProcessor(model="llama2")
        tts = HuggingFaceTTSProcessor()
        logger.info("All processors initialized")
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        return

    # Build pipeline
    pipeline = Pipeline([
        transport.input(),   # Audio from Daily room
        stt,                 # Speech to text
        llm,                 # AI response
        tts,                 # Text to speech
        transport.output(),  # Audio to Daily room
    ])

    # Create and run pipeline task
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    logger.info("Voice Assistant is ready!")
    logger.info(" Speak into your microphone; you'll hear responses on your speakers")
    logger.info("Press Ctrl+C to stop")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        await task.queue_frame(EndFrame())

if __name__ == "__main__":
    asyncio.run(main())