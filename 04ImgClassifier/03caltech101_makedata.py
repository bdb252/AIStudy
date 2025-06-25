from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

# 분류 대상 카테고리 5개 선택
caltech_dir = './caltech101/101_ObjectCategories'
categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

# 이미지 크기 지정
image_w = 64
image_h = 64
# RGB이므로 3채널의 픽셀 수 계산
pixels = image_w * image_h * 3

# 이미지 데이터와 레이블(정답)을 저장할 리스트
X = []
Y = []
# 각 카테고리마다 반복
for idx, cat in enumerate(categories):
    # 레이블은 우선 모든 크랠스에 대해 0으로 설정
    label = [0 for i in range(nb_classes)]
    # 현재 카테고리의 인덱스에 해당하는 클래스에 대해 1을 설정
    label[idx] = 1
    # 이미지 폴더 설정
    image_dir = caltech_dir + "/" + cat
    # jpg 이미지 가져오기
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        # 이미지 오픈 후 RGB 모드 변환 및 크기 조정
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        # 이미지를 numpy 배열로 변환
        data = np.asarray(img)
        # 이미지와 레이블 데이터를 리스트에 추가
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

# 리스트를 ndarray로 변환
X = np.array(X)
Y = np.array(Y)

# 학습 전용 데이터와 테스트 전용 데이터 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# npz 포맷으로 저장
np.savez("./saveFiles/caltech_5object.npz", X_train=X_train, X_test=X_test,
         Y_train=Y_train, Y_test=Y_test)
print("Task Finished", len(Y))