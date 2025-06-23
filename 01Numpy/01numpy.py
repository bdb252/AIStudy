import numpy as np
'''
ndarray
: 넘파이의 기반 데이터 타입
  다차원(Multi-dimension) 배열을 쉽게 생성하고 다양한 연산을 수행
'''

# 1차원 배열 : 3행
array1 = np.array([1,2,3])
print('타입1 : ', type(array1))
print('형태1 : ', array1.shape)

# 2차원 배열 : 2행 3열
array2 = np.array([[1,2,3],[4,5,6]])
print('형태2 : ', array2.shape)

# 2차원 배열 : 1행 3열(대괄호가 2개 있음)
array3 = np.array([[1,2,3]])
print('형태3 : ', array3.shape)

# ndim 차원표시
print('array1: {0}차원'.format(array1.ndim))
print('array2: %d차원' % array2.ndim)
print('array3: %d차원' % array3.ndim)