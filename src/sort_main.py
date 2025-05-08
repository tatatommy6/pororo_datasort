from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACE_API_KEY= os.getenv("HUGGINGFACE_API_KEY")


audio_file = "data/1.wav"
use_auth_token = "HUGGINGFACE_API_KEY"  # 실제 Hugging Face 토큰으로 교체하세요

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=use_auth_token
)

print(f"pipeline: {pipeline}")  # pipeline이 None인지 확인

if pipeline is None:
    raise RuntimeError("Pipeline 로딩에 실패했습니다. Hugging Face 토큰을 확인하세요.")

pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
diarization = pipeline(audio_file, num_speakers=2)