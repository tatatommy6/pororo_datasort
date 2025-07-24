import torch
import json
import os
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio import Audio

# Hugging Face access token 직접 입력
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=""  # 여기에 실제 토큰 입력
)

# CUDA GPU 사용
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

def diar_to_json(anno:Annotation): #Annotation 객체를 json형식으로 바꿈
    result = []
    for segment, track, label in anno.itertracks(True):
        result.append({
            "start" :segment.start,
            "end" : segment.end,
            "track" : track,
            "label" :label
        })
    
    return result

def wav_to_json(CurrentVoiceFile):
    print(f"Processing {CurrentVoiceFile}...")

    # 전체 파일에 대해 diarization 수행
    diarization = pipeline(os.path.join("separated_outputs", CurrentVoiceFile)) 
    print("전체 diarization 결과:")
    print(diarization)

    with open(f"DiarizationResultsJson/{CurrentVoiceFile[0:-4]}.json", "w") as f:
        json.dump(diar_to_json(diarization), f) # Json 저장

# # os.listdir()로 디렉토리 내의 모든 파일 이름을 리스트로 반환
for CurrentVoiceFile in os.listdir("separated_outputs"): 
    if CurrentVoiceFile.endswith("vocals.wav"):
        print(CurrentVoiceFile)
<<<<<<< HEAD
        wav_to_json(CurrentVoiceFile)    
 
#wav_to_json("VoiceWav/9.wav")
=======
        wav_to_json(CurrentVoiceFile)
            

<<<<<<< Updated upstream
#wav_to_json("VoiceWav/9.wav")
=======
wav_to_json("/Users/kimminkyeol/separated/mdx_extra_q/1/1vocals.wav")
>>>>>>> Stashed changes
>>>>>>> c06c839d6a9e0a3369bec439b44349c57c586f18
