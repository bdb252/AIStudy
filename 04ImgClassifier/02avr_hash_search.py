from PIL import Image
import numpy as np
import os, re

# 파일 경로 지정
search_dir = './caltech101/101_ObjectCategories'
cache_dir = './caltech101/cache_avhash'

# 캐시 디렉토리가 없으면 생성
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

# 이미지 데이터를 Average Hash로 변환
def average_hash(fname, size=16):
    fname2 = fname[len(search_dir): ]
    # 경로의 구분자 변경
    fname2 = fname2.replace("\\", "/")
    # 캐시파일 경로 생성(파일명의 경로를 통해 생성함)
    cache_file = cache_dir + "/" + fname2.replace('/', '_')+ '.csv'

    # 캐시 파일이 없을 경우 Average Hash 생성
    if not os.path.exists(cache_file):
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.array(img.getdata()).reshape((size, size))
        avg = pixels.mean()
        px = 1 * (pixels > avg)
        # Average Hash로 변환된 데이터를 CSV 파일로 저장한다.
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
    else:
        # 캐시 파일이 존재하면 데이터를 읽어 Numpy배열로 로드한다.
        px = np.loadtxt(cache_file, delimiter=",")
    return px

# 해밍 거리 계산
def hamming_dist(a, b):
    # 2개의 해시값을 받은 후 1차원 배열로 변환
    aa = a.reshape(1,-1)
    ab = b.reshape(1,-1)
    # 서로 다른 원소의 개수를 합산하여 해밍거리 계산
    dist = (aa != ab).sum()
    return  dist

# 주어진 디렉토리 경로에서 하위 디렉토리를 포함한 모든 파일을 순회
def enum_all_files(path):
    # 디렉토리 내의 모든 파일 검색
    for root, dirs, files in os.walk(path):
        for f in files:
            # 파일명을 얻어온 후
            fname = os.path.join(root, f)
            # 해당 확장자만 파일명을 반환
            if re.search(r'\.(jpg|jpeg|png)$', fname):
                yield fname

# 이미지 찾기
def find_image(fname, rate):
    src = average_hash(fname)
    # 검색 대상의 디렉토리의 모든 파일을 비교
    for fname in enum_all_files(search_dir):
        fname2 = fname.replace("\\", "/")
        dst = average_hash(fname2)
        # 두 이미지간의 해밍거리 계산 후 256으로 나눠서 유사도 계산
        diff_r = hamming_dist(src, dst) / 256
        # 유사도가 주어진 기준(rate) 이하인 경우 결과 반환
        if diff_r < rate:
            yield (diff_r, fname2)

# 검색할 기준 이미지 경로 설정
srcfile = search_dir + "/chair/image_0016.jpg"
# srcfile = search_dir + "/butterfly/image_0015.jpg"
html = ""
# 유사한 이미지 검색(rate : 0.25)
sim = list(find_image(srcfile, 0.25))
sim = sorted(sim, key=lambda x:x[0])
for r, f in sim:
    print(r, ">", f)
    f2 = "." + f
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>'+ \
        '<p><a href="' + f2 + '"><img src="' + f2 + '">'+ \
        '</a></p></div>'
    html += s

# HTML로 출력하기
html = """<html><head><meta charset="utf8"></head>
<body><h3>원래 이미지</h3><p>
<img src='{0}'></p>{1}
</body></html>""".format("."+srcfile, html)
# 결과를 HTML 파일로 저장하기
with open("./saveFiles/avhash-search-output.html", "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 저장 Ok")
