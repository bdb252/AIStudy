from sklearn import model_selection, svm, metrics

# csv 파일을 읽어들이고 가공하기
def load_csv(fname):
    # 레이블과 이미지 데이터를 저장할 리스트
    labels = []
    images = []

    # 파일을 읽기모드로 오픈한 후 레이블과 이미지 데이터 추가
    with open(fname, "r") as f:
        # 파일의 내용을 한줄씩 읽은 후..
        for line in f:
            # 콤마로 구분되어 있으므로 split해서 배열로 반환
            cols = line.split(",")
            # 배열의 길이가 2미만이면 정상적인 데이터가 아니므로 통과
            if len(cols) < 2: continue
            # 0번 인덱스는 레이블이므로 삭제 후 반환되는 값을 추가하고..
            labels.append(int(cols.pop(0)))
            # 나머지 부분은 이미지의 픽셀데이터를 가져와서 정규화(0~1사이 값) 시켜준다.
            vals = list(map(lambda n:int(n) / 256, cols))
            # images 리스트에 추가한다.
            images.append(vals)
    # 딕셔너리로 만들어서 반환
    return {"labels":labels, "images":images}

# 함수를 실행하여 csv 파일을 데이터로 변환
data = load_csv('./resMnist/train.csv')
test = load_csv('./resMnist/t10k.csv')

# 학습하기
clf = svm.SVC()
clf.fit(data['images'], data['labels'])

# 예측하기
predict = clf.predict(test['images'])

# 결과 확인하기
ac_score = metrics.accuracy_score(test['labels'], predict)
cl_report = metrics.classification_report(test['labels'], predict)
print('정답률 =', ac_score)
print("리포트 =")
print(cl_report)