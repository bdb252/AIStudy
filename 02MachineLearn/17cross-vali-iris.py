import pandas as pd
from sklearn import svm, model_selection

# 붓꽃의 CSV 데이터 로드
csv = pd.read_csv('resData/iris.csv')

# 리스트를 훈련 전용 데이터와 테스트 전용 데이터로 분할
data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

# 크로스 밸리데이션
clf = svm.SVC()
'''
cross_val_score() 함수는 교차 검증을 수행한다. 
cv 옵션에 지정한 수만큼 데이터를 분할하여 학습과 테스트를 반복한다.
즉 여기서는 5등분(folds)해서 5번 학습과 테스트를 진행하겠다는 의미.
'''
scores = model_selection.cross_val_score(clf, data, label, cv=5)

# 확인
print("각각의 정답률 =", scores)
print("평균 정답률 =", scores.mean())