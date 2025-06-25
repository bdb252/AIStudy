# 케라스 사용하기
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

#MNIST 데이터 로드
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 전처리
'''
데이터를 float32로 변환하고 정규화한다. 
28x28 크기의 2차원 이미지를 784차원의 1차원 배열로 변환하고
훈련용 6만개, 테스트용 1만개로 나눈다.
'''
X_train = X_train.reshape(60000,784).astype("float32")
X_test = X_test.reshape(10000,784).astype("float32")
X_train /=255
X_test /= 255

# 원-핫 인코딩 형태로 변환
'''
정수형 레이블 데이터를 0~9까지의 카테고리를 나타내는 배열로 변환한다.
즉 원-핫 인코딩 형태로 변환한다. (MNIST는 숫자를 표현한 데이터 셋) '''
y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test,10)

# 모델 구조 정의하기
# Sequential() : 모델의 각 레이어를 순차적으로 추가하는 방식의 신경망 구성
model = Sequential()
# Dense(n) : 완전연결층으로 n개의 뉴런을 가진다. 입력데이터는 784차원으로 설정
model.add(Dense(512, input_shape=(784, )))
# 활성화 함수는 ReLU로 설정
model.add(Activation('relu'))
# 과적합 방지를 위해 20%의 뉴런을 학습해서 제외함
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 여기까지 은닉층, 다음부터는 출력층으로 구성됨. 즉 2개의 은닉층과 1개의 출력층
# 10개의 뉴런 및 각 클래스(숫자0~9)에 속할 확률을 계산함
model.add(Dense(10))
# softmax 활성화 함수 사용
model.add(Activation('softmax'))

# 모델 구축
'''
손실함수 : 크로스 엔트로피
옵티마이저 : 아담
학습 중 모델의 정확도를 측정하도록 설정
'''
model.compile(
    loss = 'categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# 모델 학습
hist = model.fit(X_train, y_train)

# 테스트 데이터로 평가
score = model.evaluate(X_test, y_test, verbose=1)
print("loss =", score[0])
print("accuracy =", score[1])