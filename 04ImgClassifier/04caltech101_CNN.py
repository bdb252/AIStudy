from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

# 카테고리 지정 : 분류할 대상 클래스 목록을 5개로 설정
categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

# 이미지 크기 지정
image_w = 64
image_h = 64

# npz 파일 불러오기
data = np.load('./saveFiles/caltech_5object.npz')
# np.savez()함수로 저장시 이미 데이터를 훈련용, 테스트용으로 분리했음
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]

# 학습 및 테스트데이터 정규화
# 픽셀데이터를 0~1사이로 변환하여 신경망 학습 안정화
X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
# 데이터 형태 출력(샘플개수, 64, 64, 채널수)
print("X_train shape: ", X_train.shape)

# 모델 생성 : 레이어를 순차적으로 쌓는 방식으로 Sequential 모델 생성
model = Sequential()

# 입력층 : 첫번째 합성곱(convolution)층
'''
3*3 크기의 필터 32개를 적용하고
padding : 출력 크기를 동일하게 유지하는 same으로 설정
input_shape : 입력데이터 형태 지정(64*64*채널수)
'''
model.add(Conv2D(32, (3,3), padding="same",
                 input_shape=X_train.shape[1:]))
# 활성화함수 ReLU 적용(비선형성 추가)
model.add(Activation('relu'))
# 2*2 최대 풀링 적용
model.add(MaxPooling2D(pool_size=(2,2)))
# 과적합 방지를 위한 드롭아웃 적용(25% 뉴런 비활성화)
model.add(Dropout(0.25))

# 은닉층1 : 두번째 합성곱 층
model.add(Conv2D(32, (3,3), padding="same",
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))

# 은닉층2 : 세번째 합성곱 층
model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 은닉층3 : 완전 연결층(Fully Connected Layer)
model.add(Flatten()) # 2D 특징맵을 1D벡터로 변환
model.add(Dense(512)) # 완전 연결층(512개의 뉴런)
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 출력층 : 클래스의 개수만큼 뉴런 생성
model.add(Dense(nb_classes))
# 다중 클래스 분류를 위한 소프트맥스 활성화 함수 지정
model.add(Activation('softmax'))
# 손실함수, 옵티마이저, 평가지표
model.compile(loss='binary_crossentropy', #손실함수 지정
              optimizer='rmsprop', #RMSprop 옵티마이저 사용
              metrics=['accuracy']) #평가 지표로 정확도 사용

# 모델 훈련(학습 데이터 적용)
model.fit(X_train, Y_train, batch_size=32, epochs=50)

# 모델 평가
score=model.evaluate(X_test, Y_test)

# 손실 출력
print("loss=", score[0])
# 정확도 출력
print("accuracy=", score[1])
'''
batch_size, epochs는 딥러닝에서 사용하는 하이퍼 파라미터
batch_size : 배치크기로 전체 데이터를 한번에 모두 처리하는 대신 작은 덩어리(batch)로
    나누어서 처리한다. 즉 훈련에 사용하기 위한 데이터의 개수를 표현한다.   
epochs : 전체 데이터를 몇번 반복해서 훈련할지 지정한다.
    에폭의 수가 많으면 훈련 데이터에 대해 더 많이 학습하게 되지만, 너무 많으면
    과적합(Overfitting)이 발생할 수 있으므로 적절하게 지정해야 한다. 
'''