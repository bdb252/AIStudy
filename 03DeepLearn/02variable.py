# TensorFlow 임포트
import tensorflow as tf

# 상수 텐서 정의
a = tf.constant(120, name="a")
b = tf.constant(130, name="b")
c = tf.constant(140, name="c")

# tf.Variable 정의. 값이 변경 가능한 텐서 정의
v = tf.Variable(0, name="v")

# 합 계산 후 변수에 할당
calc_op = a + b + c
v.assign(calc_op)

# 값 출력
print(v.numpy())