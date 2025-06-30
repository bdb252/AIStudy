import i09jfood_CNN1 as jpfood
import sys, os
from PIL import Image
import numpy as np
from datetime import datetime

# 명령줄에서 파일 이름 지정
'''
이 파일은 파이참이 아닌 아나콘다 프롬프트에서 실행한다. 
현재 경로로 이동한 후 
C:\> python 현재파일면 이미지경로1 이미지경로2 ... 경로N
'''
if len(sys.argv) <= 1:
    print("i12jfood_checker.py (<파일 이름>)")
    quit()
image_size = 100
categories = ["규동", "라멘", "스시", "오코노미야끼", "가라아게"]
calories = [600,520, 400,600,350]

# 입력 이미지를 Numpy로 변환
X = []
files = []
for fname in sys.argv[1:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.array(img)
    X.append(in_data)
    files.append(fname)
X = np.array(X)

# CNN 모델 구축
model = jpfood.build_model(X.shape[1:])
model.load_weights('./saveFiles/japanese_food_model_aug.weights.h5')

# 데이터 예측
html = ""
pre = model.predict(X)
for i, p in enumerate(pre):
   y = p.argmax()
   print("+입력:", files[i])
   print("|음식이름:", categories[y])
   print("|칼로리:", calories[y])
   html += """
       <h3>입력:{0}</h3>
       <div>
         <p><img src=".{1}" width=300></p>
         <p>음식이름:{2}</p>
         <p>칼로리:{3}kcal</p>
       </div>
   """.format(os.path.basename(files[i]),
       files[i],
       categories[y],
       calories[y])


# 리포트를 HTML로 저장
html = "<html><body style='text-align:center;'>" + \
   "<style> p { margin:0; padding:0; } </style>" + \
   html + "</body></html>"


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open("./saveFiles/japanese_food_result_"+timestamp+".html", "w") as f:
   f.write(html)


print("Task Finished..!!")