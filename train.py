import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import *

if __name__ == '__main__':

    # --- 여기부터 모든 코드를 들여쓰기 합니다 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])  # MNIST 평균 및 표준편차

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    model = CNN_50_Layer(BasicBlock, [5, 5, 5, 5, 5]).to(device)
    print("MLP based on ResNet50")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련
    num_epochs = 10
    print(f"\n{num_epochs} 에포크 훈련 시작...")

    model.train()  # 훈련 모드

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 모델은 (분류 결과, 특징 맵)을 반환
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('ResNet50 기반의 MNIST 분류 모델 훈련 완료...\n')

    MODEL_SAVE_PATH = "mnist_cnn50_trained.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"훈련된 모델 가중치를 '{MODEL_SAVE_PATH}'에 저장했습니다.\n")
