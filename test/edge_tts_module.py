# -------------------- 停用 --------------------
import os
import asyncio
import uuid
from edge_tts import Communicate
from config import AUDIO_DIR

class TTS:
    def __init__(self, voice="ja-JP-NanamiNeural", rate="+15%", pitch="+20Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def generate_tts(self, text, filename):
        communicate = Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        await communicate.save(filename)

    def synthesize_to_file(self, text):
        if not os.path.exists(AUDIO_DIR):
            os.makedirs(AUDIO_DIR)

        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        asyncio.run(self.generate_tts(text, filepath))
        return "/" + filepath  # 輸出網址