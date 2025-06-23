import matplotlib.pyplot as plt
import pandas as pd
import json

# 알파벳 출현 빈도 데이터 읽기
'''
[
    {
        'labels' : ['en', 'fr', 'id'],
        'freqs' : [
            [0.xx, 0.yy, 0.zz, ... ], #en의 알파벳 빈도
            [0.xx, 0.yy, 0.zz, ... ], #fr의 알파벳 빈도
        ]
freq.json 파일은 대략 위와 같은 형태를 가지고 있음
'''
with open("lang/freq.json", "r", encoding="utf-8") as fp:
    freq = json.load(fp)

# 언어마다 계산하기
lang_dic = {}
# 각 언어(lbl)마다 빈도 데이터(fq)를 꺼낸다.
for i, lbl in enumerate(freq[0]["labels"]) :
    fq = freq[0]["freqs"][i]
    if not (lbl in lang_dic):
        lang_dic[lbl] = fq
        continue
    for idx, v in enumerate(fq):
        lang_dic[lbl][idx] = (lang_dic[lbl][idx] + v) / 2

print('lang_dic', lang_dic)

# Pandas의 DataFrame에 데이터 넣기
# chr(97)부터 chr(122)까지, 즉 a~z까지의 알파벳을 리스트로 생성
asclist = [[chr(n) for n in range(97, 97+26)]]
print('asclist', asclist)
# 이를 인덱스로 지정하여 데이터프레임을 생성
df = pd.DataFrame(lang_dic, index=asclist)

# 그래프 그리기
plt.style.use('ggplot')

# 막대그래프
df.plot(kind='bar', subplots=True, ylim=(0,0.15))
plt.savefig('./lang/lang-plot-bar.png')

# 선그래프
df.plot(kind='line', subplots=True, ylim=(0,0.15))
plt.savefig('./lang/lang-plot-line.png')

plt.show()