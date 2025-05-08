import torch

print("✅ CUDA 사용 가능:", torch.cuda.is_available())
print("🔢 CUDA 장치 수:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("🖥️ CUDA 장치 이름:", torch.cuda.get_device_name(0))
    print("🧱 PyTorch 빌드된 CUDA 버전:", torch.version.cuda)
else:
    print("⚠️ CUDA를 인식하지 못했습니다.")
