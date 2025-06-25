import tensorflow as tf
import datetime

# 로그 디렉토리 설정(현재 시간 기반 폴더 생성)
log_dir = "log_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# @tf.function을 사용하여 그래프 모드로 실행
@tf.function
def compute():
    a = tf.constant(20, name="a")
    b = tf.constant(30, name="b")
    return a * b # 곱셈연산

# 그래프 실행 및 결과 출력
result = compute()
print("계산 결과:", result.numpy())

# TensorBoard에 그래프 기록
with writer.as_default():
    tf.summary.graph(compute.get_concrete_function().graph)

print(f"TensorBoard write ok : {log_dir}")