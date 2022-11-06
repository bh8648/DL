import torch

# 문제1
print('문제1')
X = torch.FloatTensor([[[1,2,3],[4,5,6]],[[11,22,33],[44,55,66]]])
Y = torch.FloatTensor([[3],[5]])
Z = X+Y
print(X)
print(X.size())
print(Y)
print(Y.size())
print(Z)
print(Z.size())

print('='*30)


# 문제2
print('문제2')
X = torch.FloatTensor([[[1,2,3],[4,5,6]],[[11,22,33],[44,55,66]]])

# 문제2 - (a)
temp1 = X.sum() # 모든 요소의 합
temp2 = X.mean() # 모든 요소의 평균
print(temp1)
print(temp2)

# 문제2 - (b)
temp1 = X.sum(0)
temp2 = X.mean(0)
print(temp1)
print(temp2)

# 문제2 - (c)
temp1 = X.sum(1)
temp2 = X.mean(1)
print(temp1)
print(temp2)

print('='*30)


# 문제 3.
print('문제3')
X = torch.FloatTensor([1,2,3])
X = X.reshape(1,1,3)
print(X)
print(X.size())

print('='*30)


# 문제 4.
print('문제4')
X = torch.FloatTensor(range(12))
X = X.reshape(2,3,2)
print(X)

print('='*30)


# 문제 5.
print('문제5')
X = torch.FloatTensor([[[1,2,3],[4,5,6]],[[11,22,33],[44,55,66]]])
X = [X[:, :, 0].reshape(2,2,1), X[:, :, 1].reshape(2,2,1), X[:, :, 2].reshape(2,2,1)]
for x in X :
    print(x)

print('='*30)

# 문제 6.
print('문제6')
X = torch.FloatTensor([[[[1,2,3],[4,5,6]],[[2,4,6],[3,5,7]]],
                       [[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]]])

Y = torch.FloatTensor([[[[1,1],[2,2],[3,3]],[[1,1],[1,1],[1,1]]],
                       [[[2,2],[3,3],[4,4]],[[1,2],[1,2],[1,2]]]])

Z = torch.matmul(X,Y)
print(Z)






