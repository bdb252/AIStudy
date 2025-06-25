# TensorFlow 임포트
import tensorflow as tf

# 상수 텐서 정의
a = tf.constant(1234)
b = tf.constant(5678)

# 함수 정의
@tf.function #텐서플로의 그래프 모드 최적화를 적용하는 데코레이터
def add_op(a, b):
    return a+b

# 함수 호출 후 결과를 Numpy 배열로 변환하여 출력
res = add_op(a, b).numpy()
print(res)