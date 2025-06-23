# 모듈 임포트
from flask import Flask, request, render_template_string
import os
import joblib

# Flask 애플리케이션 설정
app = Flask(__name__)

# 학습된 모델 로드
pklfile = os.path.dirname(__file__) + '/lang/freq.pkl'
clf = joblib.load(pklfile)

# 텍스트 입력 양식 및 판정 결과 출력
# 라우팅 설정 '/'으로 get, post 방식으로 요청이 들어오면 이 함수를 실행한다.
@app.route('/', methods=["GET", "POST"])
def index():
    # 텍스트 입력받기
    text = request.form.get('text', '')
    msg = ''

    # 입력받은 값이 있다면 함수를 호출해서 어떤 언어인지 판단
    if text:
        lang = detect_lang(text)
        msg = "판정결과:"+lang

    # 렌더링 되는 HTML 코드를 직접 작성
    return render_template_string('''
        <html>
        <body>
            <form method="post">
                <textarea name="text" rows="8" cols="40">{{text}}</textarea>
                <p><input type="submit" value="판정"></p>
                <p>{{msg}}</p>
            </form>
        </body>
        </html>
    ''', text=text, msg=msg)
    # 입력받은 text와 어떤 언어인지 판단한 msg를 함꼐 인수로 전달한다.
    # 그러면 {{변수명}} 과 같이 사용할 수 있다.

# 알파벳 출현빈도로 언어 판별하기
def detect_lang(text):
    # 입력받은 문자열을 소문자로 변경
    text = text.lower()
    # a와 z의 아스키코드를 얻어옴
    code_a, code_z = (ord("a"), ord("z"))
    # 0으로 채워진 리스트 생성
    cnt = [0 for i in range(26)]
    # 알파벳을 확인 후 횟수 판단
    for ch in text:
        n = ord(ch) - code_a
        if 0 <= n <= 26: cnt[n] += 1
    total = sum(cnt)
    if total == 0: return "입력이 없습니다."
    freq = list(map(lambda n : n / total, cnt))

    # 언어 예측하기
    res = clf.predict([freq])

    # 언어 코드를 한국어로 변환하기
    lang_dic = {"en": "영어", "fr": "프랑스어", "id":"인도네시아어",
                "tl":"따갈로그어"}
    return lang_dic.get(res[0], "알수 없는 언어")

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)