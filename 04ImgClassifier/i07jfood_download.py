import os, re, time
import urllib.request as req
import urllib.parse as parse
import json

# 포토주 API와 캐시 디렉토리 지정
PHOTOZOU_API = "https://api.photozou.jp/rest/search_public.json"
CACHE_DIR = "./download/cache"

# 포토주 API로 이미지 검색
def search_photo(keyword, offset=0, limit=100):
    # 요청 URL 만들기
    keyword_enc = parse.quote_plus(keyword)
    q = "keyword={0}&offset={1}&limit={2}".format(keyword_enc, offset, limit)
    url = PHOTOZOU_API + "?" + q
    print("url", url)

    # 캐시 디렉토리가 없으면 생성
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # 요청주소를 기반으로 캐시 파일 이름 생성
    cache = CACHE_DIR + "/" + re.sub(r'[^a-zA-Z0-9\%\#]+', '_', url)

    # 캐시 파일이 존재하면 API 호출을 생략하고 캐시 데이터 반환
    if os.path.exists(cache):
        return json.load(open(cache, "r", encoding="utf-8"))

    # 캐시 파일이 없을 경우 API 호출
    print("[API] " + url)
    req.urlretrieve(url, cache)
    time.sleep(1) # 1초 쉬기
    return json.load(open(cache, "r", encoding="utf-8"))

# 이미지 다운로드
def download_thumb(info, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if info is None: return

    # 결과 데이터에 'photo' 키가 없으면 오류 메시지 출력 후 중단
    if not "photo" in info["info"]:
        print("[ERROR] broken info")
        return

    # 이미지 정보 리스트 추출
    photolist = info["info"]["photo"]
    for photo in photolist:
        title = photo["photo_title"] #이미지 제목
        photo_id = photo["photo_id"] #아이디
        url = photo["thumbnail_image_url"] #썸네일URL
        path = save_dir + "/" + str(photo_id) + "_thumb.jpg"

        # 이미 다운로드된 파일이 있으면 건너뜀
        if os.path.exists(path):
            continue

        # 이미지 다운로드
        try:
            print("[download]", title, photo_id)
            req.urlretrieve(url, path)
            time.sleep(1) # 다운로드 사이에 1초 대기 (서버 부하 방지)
        except Exception as e:
            print("[ERROR] 다운로드 실패:", url)

# 모두 검색하고 다운로드
def download_all(keyword, save_dir, maxphoto=1000):
    offset = 0      # 검색 결과의 시작 위치
    limit = 100     # 한 번에 검색할 결과 개수
    # 반복적으로 API 호출 및 다운로드 수행
    while True:
        # API 호출
        info = search_photo(keyword, offset=offset, limit=limit)
        # 검색 결과가 없으면 종료
        if info is None:
            print("[ERROR] 결과없음")
            return
        # 결과 데이터에 필요한 키가 없으면 오류 메시지 출력 후 종료
        if (not "info" in info) or (not "photo_num" in info["info"]):
            print("[ERROR] 키값없음")
            return
        # 검색된 총 이미지 수
        photo_num = info["info"]["photo_num"]
        if photo_num == 0:
            print("photo_num = 0, offset=", offset)
            return
        # 사진 정보가 포함돼 있으면 다운받기
        print("[download] offset=", offset)
        download_thumb(info, save_dir)
        # 다음 offset 설정
        offset += limit
        # 최대 다운로드 개수를 초과하면 종료
        if offset >= maxphoto:
            break

if __name__ == '__main__':
    # 키워드와 다운로드 폴더명을 매핑
    tasks = [
        ("牛丼", "./download/Gyudon"),
        ("ラーメン", "./download/Ramen"),
        ("寿司", "./download/Sushi"),
        ("お好み焼き", "./download/Okonomiyaki"),
        ("唐揚げ", "./download/Karaage"),
    ]
    # 반복문을 통해 함수 호출
    for keyword, folder in tasks:
        download_all(keyword, folder)

    print("Task Finished..!!")
