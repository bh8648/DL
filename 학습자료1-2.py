import torch

# 임의의 값으로 채워진 원하는 크기의 텐서 생성
x = torch.FloatTensor(3,2)
print(x)


# long 타입과 byte 타입 텐서 생성
lt = torch.LongTensor([[1,2],[3,4]])
bt = torch.ByteTensor([[1,1],[0,1]])
print(lt)
print(bt)

print('='*20)

# 텐서의 타입 변환 (casting)
lt = lt.float()
print(lt)

print('='*20)


import numpy as np

# 넘파이(numpy)배열 생성
x = np.array([[1,2],[3,4]])
print(x, type(x))


# 넘파이 배열을 파이토치 텐서로 변환
x = torch.from_numpy(x)
print(x, type(x))


# 파이토치 텐서를 넘파이 배열로 변환
x = x.numpy()
print(x, type(x))

print('='*20)

#텐서의 크기 구하기
x = torch.FloatTensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print(x.size())
print(x.shape)


#텐서의 특정차원의 크기 접근
print(x.size(0))
print(x.shape[0])
print(x.size(-1))
print(x.shape[-1])

print()

#텐서 차원의 개수
print(x.dim())
print(len(x.size()))

print('='*20)

# 텐서의 사칙연산
a = torch.FloatTensor([[1,2],[3,4]])
b = torch.FloatTensor([[2,2],[3,3]])

# 텐서의 덧셈
c = a+b
print(c)

# 텐서의 뺄셈
d = a-b
print(d)

# 텐서의 곱셈 (요소별)
e = a*b
print(e)

# 텐서의 나눗셈 (요소별)
f = a/b
print(f)

print()

# 텐서의 거듭제곱(요소별)
a = torch.FloatTensor([[1,2],[3,4]])
b = torch.FloatTensor([[2,2],[3,3]])
g = a**b
print(g)

# 텐서의 논리연산
print(a == b)
print(a != b)

print('='*20)

# 텐서의 인플레이스 (in-place) 연산 : 기존의 텐서가 연산결과로 대체됨
a = torch.FloatTensor([[1,2],[3,4]])
b = torch.FloatTensor([[2,2],[3,3]])

#기존 연산
print('<기존연산>')
print(a.mul(b))
print(a)

#인플레이스 연산
print('<인플레이스연산>')
print(a.mul_(b))
print(a)

print('='*20)

#텐서 요소의 전체 합(sum)과 평균(mean)
x = torch.FloatTensor([[1,2],[3,4]])
print(x.sum())
print(x.sum(dim=0)) #텐서 세로축(dim=0)에 대해 합(sum)연산
print(x.sum(dim=-1)) #텐서 마지막차원(dim=-1)에 대해 합(sum)연산
print(x.mean())
print(x.mean(dim=-1))

print('='*20)

# 브로드캐스트(broadcast)연산 : 텐서 + 스칼라
x = torch.FloatTensor([[1,2],[3,4]])
y = 1
z = x+y
print('텐서 + 스칼라')
print(z)
print(z.shape)
print(z.size())

print()

# 브로드캐스트(broadcast)연산 : 텐서(2D) + 벡터
x = torch.FloatTensor([[1,2],[4,8]])
y = torch.FloatTensor([3,5])
z = x+y
print('텐서 + 벡터')
print(x.size())
print(y.size())
print(z) #벡터가 각 차원에 모두 들어가서 더해짐
print(z.size())

print()

# 브로드캐스트(broadcast)연산 : 텐서(3D) + 벡터
x = torch.FloatTensor([[[1,2],[3,4]]])
y = torch.FloatTensor([3,5])
z = x+y
print('텐서(3D) + 벡터')
print(x.size())
print(y.size())
print(z) # <- 벡터가 텐서의 각 차원에 더해지며, 마지막 차원은 대응되는 위치끼리 더함.
print(z.size())

print()

# 브로드캐스트(broadcast)연산 : 텐서(2D) + 벡터(2D)
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[3],[5]])
z = x+y
print('텐서(2D) + 벡터(2D)')
print(x.size())
print(y.size())
print(z) # <- 일치하는 각 차원끼리 더해짐
print(z.size())

