import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os

# 입력 및 출력 디렉토리 설정
input_dir = "./data"  # .wav 파일이 있는 디렉토리
output_dir = "./separated_outputs"  # 분리된 음성 파일을 저장할 폴더
os.makedirs(output_dir, exist_ok=True)  # 폴더 만들기

# 사전 학습된 모델 불러오기
model = get_model(name="mdx_extra")  # 'htdemucs', 'demucs_quantized' 등 다른 모델도 가능
model = model.cuda()  # 모델을 GPU로 이동

# 모델이 올라간 디바이스 가져오기
device = next(model.parameters()).device

# 반복 처리
for i in range(9, 14):  # 9.wav부터 13.wav까지 처리
    file_name = f"{i}.wav"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.isfile(file_path):
        print(f"[경고] 파일 없음: {file_path}")
        continue

    print(f"[진행 중] 분리 중: {file_name}")

    # 파일 로드
    wav, sr = torchaudio.load(file_path)

    # 모노 → 스테레오 변환 필요 시
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # 배치 차원 추가 및 GPU로 이동
    wav = wav.unsqueeze(0).to(device)

    # 모델 적용
    with torch.no_grad():
        sources = apply_model(model, wav, split=True, overlap=0.25, progress=True)

    # 분리된 소스 저장
    sources_names = model.sources
    for idx, name in enumerate(sources_names):
        output_path = os.path.join(output_dir, f"{i}_{name}.wav")
        torchaudio.save(output_path, sources[0, idx].cpu(), sample_rate=sr)  # 저장은 CPU로
        print(f"  → 저장됨: {output_path}")
