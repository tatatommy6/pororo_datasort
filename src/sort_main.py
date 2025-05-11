import torch
import json
import os
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio import Audio

# Hugging Face access token 직접 입력
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="니꺼적어라"  # 여기에 실제 토큰 입력
)

# os.listdir()로 디렉토리 내의 모든 파일 이름을 리스트로 반환
for CurrentVoiceFile in os.listdir("VoiceWav"): 
    if CurrentVoiceFile.endswith(".wav"):
        print(f"Processing {CurrentVoiceFile}...")

        # 전체 파일에 대해 diarization 수행
        diarization = pipeline(os.path.join("VoiceWav", CurrentVoiceFile)) 
        print("전체 diarization 결과:")
        print(diarization)


# 일부 구간에 대해서도 수행
excerpt = Segment(start=2.0, end=5.0)
waveform, sample_rate = Audio().crop("2_output.wav", excerpt)
excerpt_diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
print("일부 구간 diarization 결과:")
print(excerpt_diarization)
