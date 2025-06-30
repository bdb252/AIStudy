from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 카테고리 지정 : 분류할 대상 클래스 목록을 5개로 설정
categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

# 이미지 크기 지정
image_w = 64
image_h = 64

# npz 파일 불러오기
data = np.load('./saveFiles/caltech_5object.npz')
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]

# 학습 및 테스트데이터 정규화
X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
print("X_train shape: ", X_train.shape)

# 데이터 증강 적용
train_datagen = ImageDataGenerator(
    rotation_range=10, #이미지를 -10~10도 사이로 랜덤 회전
    width_shift_range=0.1, #가로 방향으로 최대 10% 이동
    height_shift_range=0.1, #세로 방향으로 최대 10% 이동
    horizontal_flip=True, #이미지를 수평방향으로 반전(뒤집기)
    zoom_range=0.1, #최대 10%확대 or 축소(랜덤)
    fill_mode='nearest' #변환시 생긴 빈 픽셀을 가장 가까운 픽셀값으로 채움
)

# 모델 생성 : 레이어를 순차적으로 쌓는 방식으로 Sequential 모델 생성
model = Sequential()
# 입력층 : 첫번째 합성곱(convolution)층
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 은닉층1 : 두번째 합성곱 층
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation('relu'))
# 은닉층2 : 세번째 합성곱 층
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 은닉층3 : 완전 연결층(Fully Connected Layer)
model.add(Flatten()) # 2D 특징맵을 1D벡터로 변환
model.add(Dense(512)) # 완전 연결층(512개의 뉴런)
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 출력층 : 클래스의 개수만큼 뉴런 생성
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 손실함수, 옵티마이저, 평가지표
model.compile(loss='binary_crossentropy', #손실함수 지정
              optimizer='rmsprop', #RMSprop 옵티마이저 사용
              metrics=['accuracy']) #평가 지표로 정확도 사용

# 데이터 증강을 적용한 모델 훈련
'''
steps_per_epoch : 한 epoch(전체 학습 데이터 1회 학습) 당 몇번의 batch를 수행할지 지정함.
'''
model.fit(train_datagen.flow(X_train, Y_train, batch_size=32),
          steps_per_epoch=len(X_train) // 32, epochs=50)

# 모델 평가
score=model.evaluate(X_test, Y_test)
# 손실 출력
print("loss=", score[0])
# 정확도 출력
print("accuracy=", score[1])
