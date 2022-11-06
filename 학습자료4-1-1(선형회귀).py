# <선형회귀>

# 선형회귀(linear regression) :
#       벡터 입력이 주어졌을 때, 선형적 관계를 지닌 출력 벡터값을 예측하는 문제

# 선형계층을 통해 문제 해결 가능능


# <선형 회귀 모델 학습>

# N개의 입력 벡터의 출력과 타깃 벡터 사이의 손실값이 최소화 되도록
#       경사하강법을 이용해 선형 계층의 파라미터를 결정.

# 손실함수 : Mean Squared Error를 사용할 수 있음
# 경사하강법 : 더 이상 손실값이 줄어들지 않을 때까지 다음 절차를 수행


# <선형 회귀 구현>

# 필요한 라이브러리 설치
# pip install matplotlib seaborn pandas sklearn

#라이브러리 불러오기
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 보스턴 주택 가격 데이터셋 불러오기

from sklearn.datasets import load_boston
boston = load_boston()
#print(boston.DESCR) # 해당 수집 자료는 나중에 사라질 예정이라 함
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target # boston.target값으로 TARGET컬럼 생성
print(df.tail()) # 테이블의 마지막5개 행 출력


# 속성의 분포 및 속성 사이의 선형적 관계 유무 확인
#sns.pairplot(df)
#plt.show()


# TARGET 속성과 선형적 관계를 갖는 것처럼 보이는 일부 속성들에 대한 페어플롯
cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]
#sns.pairplot(df[cols])
#plt.show()


# 파이토치 불러오기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 넘파이 데이터를 파이토치 텐서로 변환

data = torch.from_numpy(df[cols].values).float()
print(data.shape)


# 데이터를 입력(x)와 출력(y)로 나누기
y = data[:, :1]
x = data[:, 1:]

print(x.shape, y.shape)


# 학습에 필요한 값 설정
n_epochs = 2000
learning_rate = 1e-3
print_interval = 100

# 모델 생성
model = nn.Linear(x.size(-1), y.size(-1))
print(model)

# 옵티마이저 생성
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for i in range(n_epochs) :
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad() # grad를 0으로 초기화.
    loss.backward() # 미분 1회 수행

    optimizer.step() # 경사하강법 1회 수행

    if (i+1) % print_interval == 0 :
        #print('Epoch {}: loss={:.4e}'.format(i+1, loss))
        print('Epoch %d: loss=%.4e'% (i+1, loss))

# 학습결과 확인
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])
sns.pairplot(df, height=5)
plt.show()

# y와 y_hat이 선형적 관계를 가짐
# 왼쪽위가 y의 분포, 오른쪽 아래가 y_hat의 분포











