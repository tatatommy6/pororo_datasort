import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os

# 입력 및 출력 디렉토리 설정
input_dir = "./data"  # .wav 파일이 있는 디렉토리
output_dir = "./separated_outputs" # 분리된 음성 파일을 저장할 폴더
os.makedirs(output_dir, exist_ok=True) #폴더 만들기

# 사전 학습된 모델 불러오기
model = get_model(name="mdx_extra") # 다른 모델: 'htdemucs','demucs_quantized'등이 있는데 보컬이 아니다 맞다 두개의 파일로 분리되주어서 편함
model.cpu() #.mps()안됌
#model.cuda() #CUDA 쓸때는 이거 쓰기  

# 반복 처리
for i in range(1, 9): # 5.wav부터 13.wav까지 파일을
    file_name = f"{i}.wav" #파일 이름 생성하고
    file_path = os.path.join(input_dir, file_name) 

    if not os.path.isfile(file_path):# 파일이 존재하지 않으면 경고 날리고
        print(f"[경고] 파일 없음: {file_path}") 
        continue

    print(f"[진행 중] 분리 중: {file_name}") #존재하면 분리 시작
    
    # 파일 로드
    wav, sr = torchaudio.load(file_path)
    
    # 모노 → 스테레오 변환 필요 시
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1) #1체널을 2번 복제하여 2체널로 변환함

    wav = wav.unsqueeze(0)
    #wav = wav.unsqueeze(0).to("cuda") #CUDA로 연산할때는 이거 쓰기

    # demucs 모델은 [batch, channels, sales] 형태로 입력을 받음
    #unsqueeze(0) 을 하면 기존 (2, samples) 형태가 (1, 2, samples)로 변환됨

    # 모델 적용
    with torch.no_grad(): #추론 모드에서 연산 수행 demucs 모델을 사전학습 모델이므로, gradient 계산이 필요 X
        sources = apply_model(model, wav, split=True, overlap=0.25, progress=True) #demucs 라이브러리의 apply_model 함수 사용

    # 분리된 소스 저장
    sources_names = model.sources #모델이 분리한 트랙 이름 리스트 가져옴
    for idx, name in enumerate(sources_names): 
        output_path = os.path.join(output_dir, f"{i}_{name}.wav")
        torchaudio.save(output_path, sources[0, idx], sample_rate=sr)  #sources[0, idx]는 분리된 특정 트랙 오디오 tensor..(?)
        print(f"  → 저장됨: {output_path}")