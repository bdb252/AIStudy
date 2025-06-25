import tensorflow as tf

# 첫번째 텐서 정의 : 정수형 상수 텐서
a = tf.constant([1,2,3], dtype=tf.int32)
# 두번째 텐서 정의 : 스칼라
b = tf.constant(2, dtype=tf.int32)
# 텐서 a와 b를 요소별로 곱셈 수행
x2_op = a * b
# 결과를 출력하기 위해 numpy() 메서드로 값을 가져옴
print(x2_op.numpy())

# 텐서 재정의
a = tf.constant([10,20,10], dtype=tf.int32)
x2_op = a * b
print(x2_op.numpy())
