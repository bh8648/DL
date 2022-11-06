# <심층신경망>

# 심층신경망(Deep Neural Network)을 사용하는 이유 :
#       비선형 적인 분포의 데이터를 모델링 하기 위해

# 심층신경망 구성을 위해 선형계층을 여러개 쌓는 다면
#       두 선형 계층의 합성은 하나의 선형 계층으로 표현 가능 하기 때문에
#       결국, 이런 방법으로는 비선형 문제를 해결할 수 없다.

# 해결방법 : 선형 계층 사이에 비선형 활성(activation)함수 끼워 넣기

# Universal Approximation Theorem :
#       심층신경망의 깊이(계층의 수)와 너비(노드의 수)를 조절하여
#       그 어떤 함수도 근사계산할 수 있다는 이론적 증명

# 적절한 깊이와 너비를 결정하는 것이 관건


# <심층신경망 학습>

# 심층심경망 학습 : 경사하강법을 이용하여 파라미터 업데이트
# 계층의 중가에 따른 학습 파라미터 증가

# 역전파 알고리즘(연쇄법칙)을 이용해 손실함수의 편미분값 계산
# 미분값이 계속해서 뒤 쪽 계층으로 전달 됨 (역전파)

# 연쇄법칙에 의한 경사하강법 살펴보기

#   1) 손실함수
#   2) 3계층 신경망
#   3) 3계층 신경망에 대한 수식

# 경사하강법
# 각 계층 파라미터에 관한 그래디언트(gradient) 계산
# 연쇄법칙


# <그래디언트 소실 문제>

# 심층신경망이 깊을 경우 최적화가 잘 되지 않는 문제 발생

#   a) 활성 함수의 도함수의 절대값이 1보다 작거나 같음
#   -> 손실함수의 미분값이 작아져 파라미터의 업데이트가 제대로 수행되지 않게 됨.

# 해결방법 : 특정 구간에서 기울기 값이 1인 비선형 활성 함수 이용
# 1) 렐루 (ReLU, Rectified Linear Unit)
# 2) 리키렐루 (Leaky ReLU)


# <심층신경망을 이용한 회귀>

# 필요한 라이브러리 불러오기
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# 보스턴 주택가격 데이터 불러오기 : 13개의 속성에 대한 506개의 샘플
from sklearn.datasets import load_boston
boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target


# 데이터 정규화 : 각 열의 범위가 모두 다르기 때문

scaler = StandardScaler()
scaler.fit(df.values[:, :-1])
df.values[:, :-1] = scaler.transform(df.values[:, :-1]).round(4)

print(df.tail())


# 심층신경망 학습에 필요한 라이브러리 불러오기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 입력(x) 및 출력(y) 텐서 구성

data = torch.from_numpy(df.values).float()
y = data[:,-1:]
x = data[:,:-1]


#학습 설정값 결정
n_epochs = 200000
learning_rate = 1e-4
print_interval = 10000


# 심층신경망 구성

class MyModel(nn.Module) :
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear1 = nn.Linear(input_dim, 3)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, 3)
        self.linear4 = nn.Linear(3, output_dim)
        #self.act = nn.ReLU()
        self.act = nn.LeakyReLU()

    def forward(self, x):
        h = self.act(self.linear1(x))
        h = self.act(self.linear2(h))
        h = self.act(self.linear3(h))
        y = self.linear4(h)
        return y

# 심층신경망 구성 : 출력결과

#model = MyModel(x.size(-1), y.size(-1))
model = nn.Sequential(
    nn.Linear(x.size(-1), 3),
    nn.LeakyReLU(),
    nn.Linear(3, 3),
    nn.LeakyReLU(),
    nn.Linear(3, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1)),
    nn.LeakyReLU()
)
print(model)


# 옵티마이저 구성
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)

# 학습진행
for i in range(n_epochs) :
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i + 1) % print_interval == 0:
        print('Epoch %d : loss=%.4e' % (i + 1, loss))

# 결과 확인
df = pd.DataFrame(torch.cat([y,y_hat], dim=1).detach().numpy(),
                  columns=['y','y_hat'])
sns.pairplot(df, height=5)
plt.show()




