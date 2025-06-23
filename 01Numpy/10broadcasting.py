import numpy as np

'''
브로드 캐스팅(Broadcasting)
: 크기가 다른 두 배열간 연산을 수행할때, 작은 배열을 연산 가능하도록
  자동으로 확장된 형태로 맞춰주는 규칙을 의미한다. 
  실제로 메모리에서 배열이 복사되는 것은 아니며, 효율적인 연산을 위해
  내부적으로 확장된 것처럼 처리된다.'''
# B(스칼라값)가 배열A의 크기에 맞춰 확장되어 계산됨
A = np.array([1,2,3])
B = 5
result = A + B
print('결과1', result)

# 2행3열과 1행3열의 배열 생성
A = np.array([[1,2,3],[4,5,6]])
B = np.array([1,2,3])
# 2행3열로 확장된 후 계산 수행
result = A + B
print('결과2\n', result)

# 2행3열과 1행2열의 배열 생성
A = np.array([[1,2,3],[4,5,6]])
B = np.array([1,2])
# 오류 : 두 배열의 마지막 차원 크기가 3과 2로 다르고, 둘 중 하나가 1이 아니므로 브로드캐스팅이 되지 않는다.
# result = A+B

# 3행3열인 2차원 배열 생성
C = np.arange(9.).reshape(3,3)
# 1차원 배열 생성
x = np.array([1., 0, 0]) # 전체가 실수로 변환됨
# 배열 x를 1행3열, 3행1열인 2차원 배열로 변환
y = x.reshape(1,3)
z = x.reshape(3,1)

# 2차원배열 + 스칼라 : 스칼라가 배열로 확장되어 연산
result = C + 10
print('결과3\n', result)

# 2차원배열 + 1차원배열 : 1차원 배열이 브로드캐스팅 되어 연산
result = C + x
print('결과4\n', result)

result = C + y
print('결과5\n', result)

result = C + z
print('결과6\n', result)

result = y + z
print('결과7\n', result)