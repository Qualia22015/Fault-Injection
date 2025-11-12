import torch
import torchvision
import torchvision.transforms as transforms
import os

from PytorchFS import FS
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_50_Layer(BasicBlock, [5, 5, 5, 5, 5]).to(device)

# 2. 저장된 가중치 파일 경로
MODEL_LOAD_PATH = "mnist_cnn50_trained.pth"

# 3. 가중치 불러오기
model.load_state_dict(torch.load(MODEL_LOAD_PATH))
print('CNN_50 로딩 완료...\n')

fault_sim = FS.FS()
fault_sim.setLayerInfo(model)
print("Pytorch Fault Simulator (FS) 초기화 및 setLayerInfo(model) 완료...\n")

# 4. 모델을 평가 모드로 설정
model.eval()
print("모델 모드 : 평가 모드")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

print("Testloader 준비 완료...\n")

print('오류 주입 및 저장 시작...\n')

TRACE_DIR = 'feature_traces'

# 테스트할 데이터 개수
TEST_LIMIT = 10000

# 저장 폴더 생성
os.makedirs(TRACE_DIR, exist_ok=True)
print(f"'{TRACE_DIR}' 폴더에 트레이스 저장을 시작합니다.")

# 카운터 초기화
count_type1 = 0
count_type2 = 0

# --- 1. Baseline 실행 (오류 주입 X) -> Type 1 생성 ---
print(f"\n--- [Phase 1] Baseline 실행 (오류 주입 없음) ---")
with torch.no_grad():
    # testloader를 순회 (i는 0부터 시작)
    for i, (inputs, labels) in enumerate(testloader):
        if i >= TEST_LIMIT:
            break  # 100개만 테스트

        inputs, labels = inputs.to(device), labels.to(device)
        label_val = labels.item()

        # 모델 실행 (오류 주입 없음)
        pred_out, trace = model(inputs)
        prediction = torch.argmax(pred_out, dim=1).item()

        # 3. 타입 분리 (Type 1)
        if prediction == label_val:
            # 예측 성공 -> Type 1
            count_type1 += 1
            # 4. 파일 저장
            filename = f"{TRACE_DIR}/{label_val}_Type1_{i}.pt"

            data_to_save = {'label': label_val,
                            'prediction' : prediction,
                            'type': 0,
                            'trace': trace.cpu()}  # 0 for Type1

            torch.save(data_to_save, filename)

        if (i + 1) % 20 == 0:
            print(f"Baseline: {i + 1}/{TEST_LIMIT} 처리 완료...")

print(f"Baseline 실행 완료. Type 1 (정상) {count_type1}개 생성.")

# --- 2. Fault Injection 실행 -> Type 2 ---
print(f"\n--- [Phase 2] Fault Injection 실행 (오류 주입) ---")
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if i >= TEST_LIMIT:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        label_val = labels.item()

        for FI_epoch in range(5) :
            for targetBit in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                handle = None
                if fault_sim:
                    try:
                        handle = fault_sim.onlineSingleLayerOutputInjection(
                            model=model,
                            NofError=1,
                            targetBit=targetBit,
                            targetLayer="random",
                            targetLayerTypes=[torch.nn.Conv2d]
                        )
                    except Exception as e:
                        print(f"FS 오류 주입 중 예외 발생 (bit {targetBit}): {e}")
                        continue

                pred_out, trace = model(inputs)
                if handle: handle.remove()
                prediction = torch.argmax(pred_out, dim=1).item()

                if prediction == label_val:
                    count_type1 += 1
                    type_str = "Type1"
                    type_val = 0

                else:
                    count_type2 += 1
                    type_str = "Type2"
                    type_val = 1

                filename = f"{label_val}_{type_str}_{targetBit}_{FI_epoch}_{i}.pt"
                file_path = os.path.join(TRACE_DIR, filename)

                data_to_save = {
                    'label': label_val,
                    'prediction': prediction,
                    'type': type_val,
                    'trace': trace.cpu()
                }

                torch.save(data_to_save, file_path)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{TEST_LIMIT}개 이미지 처리 완료...")

print("Fault Injection 실행 완료.")

print("\n--- [최종 요약] ---")

print(f"총 {TEST_LIMIT}개의 데이터 테스트 완료.")
print(f"Type 1 (최종 분류 정답): {count_type1} 개")
print(f"Type 2 (최종 분류 오답): {count_type2} 개")

print(f"생성된 트레이스 파일은 '{TRACE_DIR}' 폴더에서 확인할 수 있습니다.")
