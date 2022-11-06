import torch

# 텐서 형태 변환 : view함수
# view함수 : 텐서 요소의 총 개수는 유지한 채 shape을 바꿀 수 있음
x = torch.FloatTensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print(x.size())
print(x.view(12)) # 12 = 3*2*2

#요소를 가져오는 순서 : 사전식 순서로 가져옴
# 0 0 0 -> 0 0 1 -> 0 1 0 -> 0 1 1 -> 1 0 0 ...

# view함수 : 텐서 요소의 총 개수는 유지한 채 shape을 바꿀 수 있음
print(x.view(3,4)) # 3*4 = 3*2*2

#사전식 순서로 요소를 가져와 크기가 3*1*4인 텐서를 구성함.



#-1 활용하기 : -1이 들어간 차원의 크기를 자동 계산

print(x.view(-1)) # 0번 차원의 크기가 12로 자동 계산

print(x.view(3,-1)) # 1번 차원의 크기가 4로 자동 계산

print(x.view(-1, 1, 4)) # 0번 차원의 크기가 3으로 자동 계산


# view 함수의 변환값은 같은 메모리(저장공간)를 공유하고 있기 때문에 아래 코드에서 y가 바뀌면 x도 바뀜
y = x.view(3,4)
print(x.storage().data_ptr() == y.storage().data_ptr())

#view 대신 reshape을 사용. 용법은 그대로

print()


# 텐서 형태변환 : squeeze함수

# 크기가 1x2x2인 파이토치 텐서 생성
x = torch.FloatTensor([[[1,2],[3,4]]])
print(x.size())

#squeeze함수 : 크기가 1인 차원을 없앨 수 있음
print(x)
print(x.squeeze())
print(x.squeeze().size())

# 원하는 차원을 지정해서 없앨 수 있음.
# 단 지정한 차원의 크기가 1이 아니면 없애지 못함
print(x.squeeze(0).size()) #제대로 삭제 됨
print(x.squeeze(1).size()) #삭제가 안됨

print()


# 텐서 형태변환 : unsqueeze함수

#크기가 2x2인 파이토치 텐서 생성
x = torch.FloatTensor([[1,2],[3,4]])
print(x.size())

# 지정한 차원에 크기가 1인 차원을 추가.
print(x.unsqueeze(1).size())
print(x.unsqueeze(-1).size())
print(x.unsqueeze(2).size())

# 참고 : reshape함수를 이용해서도 가능함
print(x.reshape(2,2,-1).size())


print()


# 파이토치 텐서 자르기 : 인덱싱과 슬라이싱

#크기가 3x2x2인 파이토치 텐서 생성

x = torch.FloatTensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print(x.size())

#텐서의 0번 차원의 0번 인덱스 슬라이싱
print(x[0])

#텐서의 0번 차원의 마지막 인덱스 슬라이싱
print(x[-1])

#텐서의 1번 차원의 0번 인덱스 슬라이싱
print(x[:, 0]) # 1번 차원에 해당하는 모든 0번 인덱스

print(x[1:2, 1:, :]) # 0차원에서 1번째, 1차원에서 1번째(0번째아님), 2차원에서 전부
print(x[1:2, 1:, :].size())

print()


# split함수 : 지정한 차원이 원하는 크기가 되도록 등분함.
x = torch.FloatTensor(10,4)
splits = x.split(4, dim=0)
print(x)
print(splits)
for s in splits :
    print(s.size()) #10/4 이므로 [4,4]가 두 번, [2,4]가 한 번 나옴

print()

#chunk 함수 : 지정된 개수로 텐서를 나눔
x = torch.FloatTensor(8,4)
chunks = x.chunk(3,dim=0)
for c in chunks :
    print(c.size())

print()


# index_select함수 : 원하는 인덱스의 값만을 추출

# 3x2x2크기의 파이토치 텐서 생성
# 숫자 2와 1을 포함하는 1차원 정수형 텐서 생성
x = torch.FloatTensor([[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]])
indice = torch.LongTensor([2,1])
print(x.size())

y = x.index_select(dim=0, index=indice)
print(y)
print(y.size())

# 인덱스인 indice가 [2, 1]이므로
#     tensor의 2는 0차원 중 2번,
#     tensor의 1은 0차원 중 1번을 순서대로 가져옴


print()


# 텐서 붙이기 : cat함수

# 크기가 각각 3x3인 두 개의 파이토치 텐서 생성
x = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
y = torch.FloatTensor([[10,11,12],[13,14,15],[16,17,18]])
print(x.size(), y.size())

z = torch.cat([x,y], dim=0)
print(z)
print(z.size()) # 사이즈 = [6,3]

# 차원을 -1로 지정함으로써 마지막 차원으로 두 텐서 붙이기
z = torch.cat([x,y], dim=1)
print(z)
print(z.size()) # 사이즈 = [3,6]

# 붙이고자 하는 차원 이외의 차원의 크기가 서로 맞지 않으면 붙일 수 없음.
# 예) 1번 차원으로 두 2D 텐서를 붙이려 할 때, 나머지 차원의 크기가 다른 경우

print()



# 텐서 붙이기 : stack함수
# stack: 함수 : 텐서를 샇아 더 높은 차원의 텐서를 생성

#크기가 각각 3x3인 두 개의 파이토치 텐서 생성
x = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
y = torch.FloatTensor([[10,11,12],[13,14,15],[16,17,18]])
print(x.size(), y.size())

# 3x3크기의 두 2차원 텐서를 쌓아 2x3x3 크기의 3차원 텐서를 생성
z = torch.stack([x,y])
print(z)
print(z.size())


#쌓고자 하는 차원을 지정할 수 있음
z = torch.stack([x,y], dim=-1)
print(z)
print(z.size())




