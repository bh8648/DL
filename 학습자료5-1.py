# <파라미터 업데이트 과정>

# 문제점 : 
#       1) 메모리 한계로 인해 큰 데이터셋을 한 번에 계산하기 어려움
#       2) 계산량이 많아 학습 속도가 느려짐

# 해결방법 : 확률적 경사하강법 (Stochastic Gradient Descent, SGD)


# <확률적 경사하강법>

# 전체 데이터셋 중 임의의 k개의 샘플(미니배치, mini-batch)을 비복원 추출하여
#       파라미터 업데이트 하는 방식

# 손실 함수의 최소점을 찾는 과정 : 각 화살표는 그래디언트를 나타냄
# 미니배치가 커질 수록 실제 그래디언트와 비슷해질 확률이 증가


# <확률적 경사하강법 활용>

# 필요한 라이브러리 불러오기

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


# 미국 캘리포니아 주택 가격 데이터 (9개 속성, 20640개 샘플) 불러오기
#pd.set_option('display.max_columns', None)
california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)
df["TARGET"] = california.target
print(df.tail())


# # 임의 추출된 1000개 샘플에 대한 페어플롯
# sns.pairplot(df.sample(1000))
# plt.show()


# 데이터 정규화 : 평균이 0, 표준편차가 1이 되도록
scaler = StandardScaler()
scaler.fit(df.values[:, :-1])
df.values[:, :-1] = scaler.transform(df.values[:, :-1])

# sns.pairplot(df.sample(1000))
# plt.show()


# 학습을 위한 라이브러리 불러오기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 데이터를 파이토치 텐서로 변환 및 크기 확인

data = torch.from_numpy(df.values).float()
print(data.shape) # torch.Size([20640, 9])


# 입력(x), 출력(y) 데이터 분리 및 크기 확인
x = data[:, :-1]
y = data[:, -1:]
print(x.shape, y.shape) # torch.Size([20640, 8]) torch.Size([20640, 1])


# 학습에 필요한 설정값 지정

n_epochs = 4000
batch_size = 256
learning_rate = 1e-2
print_interval = 200


# 심층신경망 구성

model = nn.Sequential(
    nn.Linear(x.size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1)),
)
print(model)

optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# 학습 진행 : 중첩 for 반복문을 이용

for i in range(n_epochs) :
    #Shuffle the index to feed-forward.
    indices = torch.randperm(x.size(0))
    # torch.index_select => # 셔플된 인덱스를 데이터에 적용시킴
    x_ = torch.index_select(x, dim=0, index=indices)
    y_ = torch.index_select(y, dim=0, index=indices)

    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)

    y_hat = []
    total_loss = 0

    for x_i, y_i in zip(x_,y_) :
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += float(loss) # This is very important to prevent memory leak

        y_hat += [y_hat_i]

    total_loss = total_loss/len(x_)
    if (i+1) % print_interval == 0 :
        print('Epoch %d : loss=%.4e' % (i+1, total_loss))

y_hat = torch.cat(y_hat, dim=0)
y = torch.cat(y_, dim=0)


# 학습 결과 확인

df = pd.DataFrame(torch.cat([y,y_hat], dim=1).detach().numpy(),
                  columns=['y','y_hat'])
sns.pairplot(df, height=5)
plt.show()





