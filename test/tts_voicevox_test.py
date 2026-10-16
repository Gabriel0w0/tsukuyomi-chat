import requests, json

# 用 /speakers 查出全部支援的 speaker 和 style
SPEAKER_ID = 58  # 猫使ビィ・人見知り

# Step 1: 取得 audio_query
resp = requests.post(
    "http://localhost:50021/audio_query",
    params={"text": "こんにちは、ご主人さま♡", "speaker": SPEAKER_ID},
)
audio_query = resp.json()

# Step 2: 合成 audio
synth_resp = requests.post(
    "http://localhost:50021/synthesis",
    params={"speaker": SPEAKER_ID},
    headers={"Content-Type": "application/json"},
    data=json.dumps(audio_query)
)

# Step 3: 儲存為 .wav 檔
with open("output.wav", "wb") as f:
    f.write(synth_resp.content)

# for Linux, MacOS
# docker run -d --name voicevox_engine -p 50021:50021 voicevox/voicevox_engine 

# for Windows
# docker pull voicevox/voicevox_engine:nvidia-latest
# docker run --rm --gpus all -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:nvidia-latest 