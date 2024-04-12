import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx, :-1], dtype=torch.float)
        target = torch.tensor(self.data[idx, -1], dtype=torch.float)
        return sample, target

# 신경망 모델 정의
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 하이퍼파라미터 설정
input_size = 16
hidden_size = 64
output_size = 3
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# 데이터 준비
# 여기에 데이터를 불러와서 적절히 전처리하여 numpy 배열로 준비해주세요
# 예시로 랜덤 데이터를 생성하겠습니다.
import numpy as np
data = np.loadtxt('data\\homework-large.txt')

# 데이터셋 및 데이터로더 생성
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = Model(input_size, hidden_size, output_size)

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        # Forward pass
        outputs = model(batch_inputs)

        print(batch_inputs.shape, batch_targets.type())
        #prediction = torch.argmax(outputs, dim=1)
        #print(prediction)
        loss = criterion(outputs, batch_targets.type(torch.long))

        # Backward pass 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 학습된 모델을 사용하여 테스트
# 여기에 테스트 데이터를 넣어서 결과를 확인할 수 있습니다.

test_data = np.loadtxt('data\\homework.txt')

# 데이터셋 및 데이터로더 생성
test_dataset = CustomDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 평가
model.eval()  # 모델을 평가 모드로 설정
correct = 0
total = 0
with torch.no_grad():  # 그래디언트 계산 비활성화
    for batch_inputs, batch_targets in test_dataloader:
        # Forward pass
        outputs = model(batch_inputs)
        #predictions = torch.round(outputs)  # 이진 분류를 위해 출력값을 반올림하여 0 또는 1로 변환
        predictions = torch.argmax(outputs,dim=1)
        correct += (predictions == batch_targets).sum().item()
        total += batch_inputs.size(0)

# 전체 테스트 데이터에 대한 예측 정확도 계산
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')