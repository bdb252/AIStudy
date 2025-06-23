import numpy as np

'''
ndarray에 저장하는 데이터는 숫자, 문자열, boolean등 모두 가능하다.
단 연산의 특성상 같은 자료형이어야 한다.'''

# 정수만으로 리스트 생성
list1 = [1,2,3]
# 리스트를 ndarray로 변환
array1 = np.array(list1)
print('array1 print:',array1)
# 배열의 타입확인 : ndarry
print('array1 type:', type(array1))
# 배열에 저장된 데이터 타입 확인 : int64
print('array1 dtype:',array1.dtype)

# 실수가 포함된 리스트 생성
list2 = [1,2,3.0]
# 배열로 변환시 모든 데이터가 실수로 변환된다.
array2 = np.array(list2)
print('array2 print:',array2)
print('array2 type:', type(array2))
# 실수가 포함되었으므로 float64로 출력된다.
print('array2 dtype:',array2.dtype)

# 문자열이 포함된 리스트 생성
list3 = [1,2,'test']
array3 = np.array(list3)
# 모든 데이터가 문자열로 변환되어 출력된다.
print('array3 print:',array3)
print('array3 type:', type(array3))
# <U21 로 출력됨. 유니코드 문자열 데이터 타입을 의미함.
print('array3 dtype:',array3.dtype)

# astype() : ndarray의 데이터 타입을 변경하는 함수
# 정수형을 실수형으로 변경
array_int1 = np.array([1,2,3])
array_float1 = array_int1.astype('float64')
print(array_float1,array_float1.dtype)

# 실수형을 정수형으로 변경
array_float2 = np.array([1.1,2.1,3.1])
array_int2 = array_float2.astype('int32')
print(array_int2, array_int2.dtype)