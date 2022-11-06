# 필요한 라이브러리 불러오기

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


# 데이터셋 불러오기

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df["TARGET"] = california.target
print(df.tail())


# 데이터 정규화
scaler = StandardScaler() # 평균이 0이고 표준편차가 1인 분포를 만드는 것이 StandardScaler다.
scaler.fit(df.values[:,:-1])
df.values[:, :-1] = scaler.transform(df.values[:,:-1])
print(df.tail())


# 학습에 필요한 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 입출력 데이터셋 구성
data = torch.from_numpy(df.values).float()
x = data[:, :-1] # 모든 행, 마지막열 제외
y = data[:, -1:] # 모든 행, 마지막열만
print(x.shape, y.shape) # torch.Size([20640, 8]) torch.Size([20640, 1])


# 학습에 필요한 설정값 지정
n_epochs = 4000
batch_size = 256
print_interval = 200
# learning_rate = 1e-2
# 이번에는 학습률을 주석처리 함


print(x.size(-1))
# 모델(신경망) 생성
model = nn.Sequential(
    nn.Linear(x.size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1))
)


# 옵티마이저 설정
# we don't need learning rate hyper-parameter.
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=0.1)


#학습진행

for i in range(n_epochs) :
    # Shuffle the index to feed-forward.
    indices = torch.randperm(x.size(0))
    x_ = torch.index_select(x, dim=0, index=indices) # 학습데이터의 인덱스 순서를 섞음
    y_ = torch.index_select(y, dim=0, index=indices) # 학습데이터의 인덱스 순서를 섞음

    x_ = x_.split(batch_size, dim=0) # 미니배치 단위로 끊어서 데이터를 저장.
    y_ = y_.split(batch_size, dim=0) # 미니배치 단위로 끊어서 데이터를 저장.

    y_hat = []
    total_loss = 0

    for x_i, y_i in zip(x_,y_) :
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        optimizer.zero_grad() # 옵티마이저에 그래디언트값 제로세팅
        loss.backward() # 미분 계산

        optimizer.step() # 경사하강법을 위에서 세팅한 optimizer로 적용

        total_loss += float(loss) # 손실함수를 실수로 변환해서 total_loss에 계속 더해줌

        y_hat += [y_hat_i] # 출력값을 계속해서 저장

    total_loss = total_loss/len((x_)) # 모든 미니배치의 계산이 끝나면 total_loss의 평균을 내준다.
    if (i+1) % print_interval == 0 :
        print('Epoch %d : loss=%.4e' % (i+1, total_loss))

y_hat = torch.cat(y_hat, dim=0)
y = torch.cat(y_, dim=0)


#학습결과 확인
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])
sns.pairplot(df, height=5)
plt.show()








