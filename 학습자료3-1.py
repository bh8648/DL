import torch

# 선형 계층(linear layer)은
#       입력 값들에 대한 가중합으로 출력을 반환하는 신경망의 기본 요소

# 선형 계층의 작동 방식 :
#       입력 노드 값들의 가중합으로 출력 노드의 값들이 결정

# 선형 계층의 연산을 행렬의 곱과 합으로 표현할 수 있음음

# 여러 개의 입력 벡터를 처리하기 위해
#       미니 배치 (mini-batch)행렬을 구성하여 한꺼번에 계산할 수 있음

# 행렬의 곱셈과 벡터의 덧셈으로 이루어져 있으므로
#       선형 변환(linear transform)으로 볼 수 있음


# <선형 계층 구현>

# 방법 1) 선형 계층 직접 구현 : matmul 함수 및 브로드캐스트 기능 이용
def linear(x,W,b) :
    y = torch.matmul(x,W)+b
    return y

print('방법1')
W = torch.FloatTensor([[1,2],[3,4],[5,6]])
b = torch.FloatTensor([2,2])
x = torch.FloatTensor(4,3) # 4개의 벡터로 구성된 입력 미니배치
y = linear(x,W,b)
print(y.size())


# 방법 2) 파이토치 클래스를 이용한 직접 구현 : nn 패키지 이용.
import torch.nn as nn

class MyLinear(nn.Module) :
    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__()
        self.W = torch.FloatTensor(input_dim, output_dim)
        self.b = torch.FloatTensor(output_dim)

    def forward(self, x):
        y = torch.matmul(x, self.W) + self.b

        return y

print('\n방법2')
x = torch.FloatTensor(4,3) # 4개의 벡터로 구성된 입력 미니배치
linear = MyLinear(3,2)
y = linear(x)
print(y.size())


# 방법3) 가중치(파라미터, parameter)를 파이토치에서 학습 가능하도록 구현 :
#        nn.Parameter 활용

class MyLinear(nn.Module) :
    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.b = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x):
        y = torch.matmul(x, self.W) + self.b

        return y

# 파라미터 확인
print('\n방법3')
x = torch.FloatTensor(4,3) # 4개의 벡터로 구성된 입력 미니배치
linear = MyLinear(3,2)
y = linear(x)
print(y.size())
for p in linear.parameters() :
    print(p)


#방법4) torch.nn에 미리 정의된 선형 계층 불러다 쓰기
print('\n방법4')
x = torch.FloatTensor(4,3)
linear = nn.Linear(3,2)
y = linear(x)
print(y.size())


# 방법 5) 클래스 내부에 선형 계층 저장하기
class MyLinear(nn.Module) :
    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.linear(x)

        return y

print('\n방법5')

x = torch.FloatTensor(4,3)
linear = nn.Linear(3,2)
y = linear(x)
print(y.size())







