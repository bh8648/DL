# <로지스틱 회귀>

# 로지스틱 회귀(logistic regression) :
#       데이터가 어떤 범주에 속할 확률에 따라 분류하는 알고리즘
# 예) 어떤 사람의 키와 몸무게가 주어졌을 때, 남자인지 여자인지 맞추는 문제

# 로지스틱 회귀 문제를 풀기 위해서는
#       출력값을 확률 값으로 만들어주는 활성 함수(activation function)가 필요


# 대표적인 활성함수 :
#   a) 시그모이드 (함수값이 0과 1 사이)
#   b) 하피어볼릭탄젠트 (함수값이 -1과 1 사이)


# <로지스틱 회귀 모델 구조>

# 선형계층의 출력값에 시그모이드 함수 적용 :
#       모델의 출력값이 0과 1 사이가 됨.


# <로지스틱 회귀 모델 학습>

# 선형 회귀 모델의 학습과 비슷한 방식이나 다른 손실함수 사용


# <로지스틱 회귀의 의미>

# 로지스틱 회귀는 분류 문제에 가까움

# 문제    출력y              예시
# 회귀    실수값(연속)        신상 정보가 주어지면 연봉 예측하기
# 분류    카테고리값(이산)     신상 정보가 주어지면 성별 예측하기

# 활성 함수의 값(0과 1 사이)에 대해 0.5를 기준으로 분류


# <로지스틱 회귀 손실함수>

# 이진 크로스엔트로피(Binary Cross-Entropy) 손실함수


# <로지스틱 회귀 수식 표현>

# 손실함수를 최소화하는 파라미터 찾기
# 경사하강법
# 모델의 출력은 "입력 샘플이 j번째 항목에 속할 확률"



# <로지스틱 회귀 구현>


# 필요한 라이브러리 불러오기

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 데이터셋(유방암) 불러오기 : 각 10개의 속성에 대한 평균, 표준편차, 최악 값들

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.DESCR)
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

# # 평균값과 class의 상관관계 파악을 위한 페어플롯
# sns.pairplot(df[['class']+list(df.columns[:10])])
# plt.show()
#
# # 표준편차 값과 class의 상관관계 파악을 위한 페어플롯
# sns.pairplot(df[['class']+list(df.columns[10:20])])
# plt.show()
#
# # 최악 값과 class의 상관관계 파악을 위한 페어플롯
# sns.pairplot(df[['class']+list(df.columns[20:30])])
# plt.show()


#히스토그램 플롯

cols = ["mean radius", "mean texture",
        "mean smoothness", "mean compactness", "mean concave points",
        "worst radius", "worst texture",
        "worst smoothness", "worst compactness", "worst concave points",
        "class"]

# for c in cols[:-1] :
#     sns.histplot(df, x=c, hue=cols[-1], bins=50, stat='probability')
#     plt.show()



# 데이터셋 설정
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df[cols].values).float()

print(data.shape) # torch.Size([569, 11])
x = data[:, :-1]
y = data[:, -1:]
print(x.shape, y.shape) # torch.Size([569, 10]) torch.Size([569, 1])


# Define configuration (환경 정의)
n_epochs = 200000
learning_rate = 1e-2
print_interval = 10000


# 모델정의

class MyModel(nn.Module) :
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear = nn.Linear(input_dim,output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.act(self.linear(x))

        return y


# 모델 생성, 손실함수 및 옵티마이저 설정

model = MyModel(input_dim=x.size(-1),
                output_dim=y.size(-1))

crit = nn.BCELoss() #Define BCELoss instead of MSELoss.

optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)

#학습

for i in range(n_epochs) :
    y_hat = model(x)
    loss = crit(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i+1) % print_interval == 0 :
        print('Epoch %d : loss=%.4e' % (i+1, loss))


#결과 확인
correct_cnt = (y == (y_hat>.5)).sum()
total_cnt = float(y.size(0))
print('Accuracy: %.4f' % (correct_cnt/total_cnt))


# 결과 값 분포 확인

df = pd.DataFrame(torch.cat([y,y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])
sns.histplot(df, x='y_hat', hue='y', bins=50, stat='probability')
plt.show()

