import torch

# expand 함수 : 크기가 1인 차원을 원하는 크기로 늘려줌
print('expand')
x = torch.FloatTensor([[[1,2]], [[3,4]]])

print(x.size()) # torch.Size([2, 1, 2])

y = x.expand(2,3,2) # 1번 차원의 크기가 3으로 늘어남
print(y) # 텐서의 성분변화. 본래 [[1,2]] -> expand [[1,2],[1,2],[1,2]]

print(y.size()) # torch.Size([2, 3, 2])


#cat 함수를 이용해 마찬가지의 결과를 얻을 수 있다.
y = torch.cat([x]*3, dim=1)
print(y)


print('='*50)

# randperm 함수 : random permutation, 즉, 임의의 순서로 정수 수열을 생성함.
print('randperm')

x = torch.randperm(10) # 0부터 9까지 임의의 정수 수열 생성
print(x)
print(x.size()) # torch.Size([10])

print('='*50)

# argmax 함수 : 텐서의 가장 큰 성분의 인덱스를 반환.
print('argmax')

# 집합 x의 원소 중, 주어진 함수 f의 값이 최대가 되는 x값을 말함
# 차원 지정시, 지정한 차원을 제외하고 차원이 같은 값들 중 가장 큰 성분의 인덱스 반환

x = torch.randperm(3**3).reshape(3,3,-1)
print(x)
print(x.size()) # torch.Size([3, 3, 3])

y = x.argmax(dim=-1)
print(y)
print(y.size()) # torch.Size([3, 3])

print('='*50)

# topk 함수 : 텐서의 성분 중 가장 큰 k개의 값과 인덱스를 반환
print('topk')
x = torch.randperm(3**3).reshape(3,3,-1)
print(x)
print(x.size())

# 가장 큰 값 하나 + 인덱스
values, indices = torch.topk(x, k=1, dim=-1)
print(values)
print(indices)
print(values.size())
print(indices.size())

# k = 2인 경우
# 가장 큰 값과 그 다음으로 큰 값 + 인덱스들
_, indices = torch.topk(x, k=2, dim=-1)
print(_)
print(indices)
print(indices.size())

print('='*50)


# sort 함수 : 지정한 차원을 따라 오름차순으로 정렬한 텐서의 성분과 인덱스를 반환.
print('sort')

x = torch.randperm(3**3).reshape(3,3,-1)
print(x)
values, indices = torch.sort(x, dim=-1)
print(values)
print(indices) # 본래 텐서 성분의 위치를 순차적으로 나열한 인덱스

print('='*50)


# masked_fill 함수 : 텐서 내의 원하는 부분을 특정 값으로 변경
print('masked_fill')

x = torch.FloatTensor([i for i in range(3**2)]).reshape(3,-1)
print(x)
print(x.size())
mask = x>4
print(mask)

y = x.masked_fill(mask, value=-1) # 해당 조건(mask)이 True인 값들을 value로 채움
print(y)

print('='*50)


# ones 함수 : 1로 채워진 주어진 크기의 텐서 생성
# zeros 함수 : 0으로 채워진 주어진 크기의 텐서 생성
print('ones & zeros')

print(torch.ones(2,3))
print(torch.zeros(2,3))

print('='*50)


# ones_like 함수 : 특정 텐서와 같은 크기의 1로 채워진 텐서 생성
# zeros_like 함수 : 특정 텐서와 같은 크기의 0로 채워진 텐서 생성
print('ones_like & zeros_like')
x = torch.FloatTensor([[1,2,3],[4,5,6]])
print(x.size())
print(torch.ones_like(x))
print(torch.zeros_like(x))

print('='*50)


# 행렬 곱
# 행렬 A와 B의 곱 : A의 열의 개수와 B의 행의 개수가 같아야 함.

# 벡터 v와 행렬 M의 곱 : 행렬 곱과 마찬가지 방식으로 연산
# 벡터는 1번 차원의 크기가 1인 행렬로 취급될 수 있음.

print('\n행렬곱\n')

# matmul 함수를 이용해 행렬 곱 연산
print('matmul')

x = torch.FloatTensor([[1,2],[3,4],[5,6]])
y = torch.FloatTensor([[1,2],[1,2]])
print(x.size(), y.size())
z = torch.matmul(x,y)
print(z)
print(z.size()) # torch.Size([3, 2])

print('='*50)


# 배치 행렬 곱 : 여러 행렬 곱을 동시에 수행하는 것, bmm 함수 이용.

# bmm 함수는 텐서의 마지막 두 차원을 행렬로 취급하여 행렬 곱 연산을 수행하므로
#       나머지 차원의 크기는 동일해야 함

print('\n배치 행렬곱\n')
print('bmm')
x = torch.FloatTensor(3,3,2) # 3x2 크기의 행렬 3개로 볼 수 있음
y = torch.FloatTensor(3,2,3) # 2x3 크기의 행렬 3개로 볼 수 있음
z = torch.bmm(x,y)
print(z.size())

print('='*50)






