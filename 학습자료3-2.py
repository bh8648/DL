import torch

# <손실(Loss)함수>

# 데이터 D를 이용해 학습된 모델 f(예: 선형계층)의 출력이 원하는 값과 얼마나 다른지 측정
# 손실함수의 값이 최소화 되도록 모델(선형계층)의 파라미터를 변경하려는 목적

# <손실(Loss)함수의 종류>

#   1) L1 노름(norm)
#   2) L2 노름(norm)
#   3) 평균 제곱 오차 (Mean Squared Error)
#   4) 제곱근 평균 제곱 오차 (Root Mean Squared Error)
#   5) MSE(평균 제곱 오차)를 이용해 정의된 손실함수


# MSE 파이토치 구현

# N : 벡터의 차원
# n : 데이터 샘플의 개수

def mse(x_hat, x) :
    y = ((x-x_hat)**2).mean()

    return y

print('MSE 구현\n')
x = torch.FloatTensor([[1,1],[2,2]])
x_hat = torch.FloatTensor([[0,0],[0,0]])
print(mse(x_hat,x)) # tensor(2.5000)

print('='*50)

# torch.nn.functional 이용
print('\ntorch.nn.functional 이용\n')

import torch.nn.functional as F
a = F.mse_loss(x_hat, x)
b = F.mse_loss(x_hat, x, reduction='sum') # 평균오차제곱인데 오차의 제곱의 합 반환
c = F.mse_loss(x_hat, x, reduction='none') # 오차의 제곱만 하고 그대로 반환
print(a)
print(b)
print(c)

# reduction : 차원 감소 연산 방법 지정 (sum, none 등)

print('='*50)


# torch.nn 이용
print('\ntorch.nn 이용\n')
import torch.nn as nn
mse_loss = nn.MSELoss()
print(mse_loss(x_hat, x)) # tensor(2.5000)

# nn.Module 하위클래스에서 하나의 계층으로 취급가능

print('='*50)


# <미분>

# 손실함수의 최소값을 구하기 위한 수치적 방법
# 기울기
# 두 점 사이의 기울기


# 미분 : 순간 변화율, 즉, x변화량을 0으로 접근 시킬 경우 얻어짐.

# 편미분 : 다변수 함수의 특정 변수에 대한 미분


# <함수의 입출력 형태>

# 입력이 벡터 또는 행렬인 함수
# 출력이 벡터 또는 행렬인 함수
# 입력과 출력이 벡터인 함수
# 입력이 벡터, 출력이 스칼라인 다변수 함수의 미분
# 입력이 행렬, 출력이 스칼라인 다변수 함수의 미분
# 입력이 스칼라, 출력이 벡터인 함수의 미분
# 입력이 벡터, 출력이 벡터인 함수의 미분



# <경사하강법 (gradient descent)>

# 손실함수의 값이 최소가 되도록 파라미터를 찾는 방법
# 기본 아이디어 : 손실함수의 도함수가 0이되는 파라미터 값 찾기>

# 경사하강법은 손실함수의 도함수 값이 0이되는 위치(x)를 수치적으로 찾는 방법
# 극점찾기

# 전역(global) 최소점과 지역(local)최소점


# <경사하강법 구현>
print('경사하강법 구현\n')

import torch
import torch.nn.functional as F

target = torch.FloatTensor([[.1,.2,.3],
                            [.4,.5,.6],
                            [.7,.8,.9]])
# rand_like 함수는 대상의 크기와 똑같은 텐서를 복사하는 함수. (내용물은 난수)
x = torch.rand_like(target)
x.requires_grad = True # 이걸 통해 중간에 계산된 기울기(미분값)을 얻을 수 있다.
print(x)
loss = F.mse_loss(x, target)
print(loss)

threshold = 1e-5 #한계점
learning_rate = 1.
iter_cnt = 0

while loss > threshold :
    iter_cnt += 1
    loss.backward() # Calculate gradients.

    x = x-learning_rate*x.grad

    x.detach_()
    x.requires_grad_(True)

    loss = F.mse_loss(x, target)

    print('%d-th Loss : %.4e' % (iter_cnt, loss))
    print(x)


print('='*50)


# 파이토치 오토그래드
# requires_grad 속성이 True인 텐서의 연산을 추적하기 위한 계산 그래프가 구축되고,
#       backward함수가 호출되면 이 그래프를 따라 미분을 자동으로 수행

x = torch.FloatTensor([[1,2],[3,4]]).requires_grad_(True)

x1 = x+2
print(x1)

x2 = x-2
print(x2)

x3 = x1*x2
print(x3)

y = x3.sum()
print(y)

y.backward()
print(x.grad) # y













