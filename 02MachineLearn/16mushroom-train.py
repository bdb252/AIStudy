# 랜덤 포레스트 사용을 위한 모듈 임포트
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 데이터프레임으로 변환
mr = pd.read_csv('./resData/mushroom.csv', header=None)
# 데이터 내부의 기호를 숫자로 변환
label = []
data = []
attr_list = []
for row_index, row in mr.iterrows():
    # 첫번째 데이터는 라벨로 사용
    label.append(row.loc[0])
    row_data = []
    for v in row.loc[1:]:
        # 각 문자를 유니코드 값(정수)로 변환
        row_data.append(ord(v))
    data.append(row_data)
    # 유니코드로 변환된 훈련데이터 확인하기(첫번째)
    if row_index == 0:
        print('row_data', row_data)
# 라벨 데이터 확인
print('label', label)

# 학습 전용 데이터와 테스트 전용 데이터로 나누기
data_train, data_test, label_train, label_test = train_test_split(data, label)

# 데이터 학습하기
clf = RandomForestClassifier()
clf.fit(data_train, label_train)

# 예측하기
predict = clf.predict(data_test)

# 결과 테스트하기
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답률 =", ac_score)
print("리포트 = \n", cl_report)