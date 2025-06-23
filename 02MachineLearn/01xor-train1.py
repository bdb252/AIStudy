# 사이킷런의 svm 모듈 임포트
from sklearn import svm

'''
XOR의 계산 결과 데이터
베타적 논리합인 XOR은 두 입력값이 동일하면 False, 다르면 True가 된다.
이 4가지의 논리를 데이터로 정의한다.
'''
xor_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

# 학습을 위해 데이터와 레이블 분리하기
data = []  # 학습용 데이터 저장용 리스트
label = [] # 레이블 데이터 저장용 리스트
for row in xor_data:
    # 앞의 2개는 학습 데이터로 사용
    p = row[0]
    q = row[1]
    # 마지막은 레이블 데이터로 사용
    r = row[2]
    # 각 리스트에 추가
    data.append([p, q])
    label.append(r)

'''
분류기(Classifier) 객체 생성
SVC(Support Vector Classification)
: 서포트 벡터 머신 분류 모델을 생성하는 클래스로, 분류 작업에 적합한
  SVM 모델 객체를 사용한다. 이 객체를 통해 데이터를 학습(fit)하고 
  예측(predict)을 수행할 수 있다. '''
clf = svm.SVC()
# 데이터 학습시키기
clf.fit(data,label)

prob = [
    [0,1],
    [1,0],
    [0,0],
    [0,0],
    [1,1],
    [1,0],
    [0,1],
    [0,1]
]
solution = [
    1,1,0,0,0,1,1,1
]
'''
데이터 예측하기 : 예측하고 싶은 데이터를 전달하면 데이터 수만큼 예측 결과를 리턴해준다.'''
pre = clf.predict(prob)
print('예측결과:',pre)

ok = 0; total = 0
# 레이블 데이터를 통해 반복
for idx, answer in enumerate(solution):
    # 예측결과와 하나씩 비교
    p = pre[idx]
    if p == answer:
        ok += 1
    total += 1

print('정답률:',ok,'/',total,'=',ok/total)