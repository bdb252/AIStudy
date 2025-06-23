from  sklearn import svm, metrics
from  sklearn.model_selection import train_test_split
import pandas as pd

csv = pd.read_csv('./resData/iris.csv')

csv_data = csv[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
csv_label = csv['Name']

# 학습데이터와 테스트데이터로 분리
train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

# 학습 및 예측
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# 정답률
ac_score = metrics.accuracy_score(test_label, pre)
print('정답률 = %.2f' % (ac_score*100)+'%')