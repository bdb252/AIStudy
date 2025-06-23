# 사이킷런, 판다스 모듈 임포트
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# 키와 몸무게 데이터 읽어들이기
tbl = pd.read_csv('resData/bmi.csv')

# 컬럼(열)을 자르고 정규화하기
label = tbl['label']
# 0~1사이의 값으로 정규화
w = tbl['weight'] / 100
h = tbl['height'] / 200
# 정규화된 몸무게와 키를 데이터프레임으로 변환
wh = pd.concat([w,h], axis=1)

# 학습 전용 데이터와 테스트 전용 데이터로 나누기
data_train, data_test, label_train, label_test = train_test_split(wh, label)

# 데이터 학습하기
clf = svm.SVC()
clf.fit(data_train, label_train)

# 예측하기
predict = clf.predict(data_test)

# 결과 테스트하기
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답률 =", ac_score)
print("리포트 = \n", cl_report)