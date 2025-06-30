import cv2, sys, re

# 입력 파일 지정(아나콘다 프롬프트에서 실행해야 함)
# CMD > python 예제파일.py 이미지경로.jpg
if len(sys.argv) <= 1:
    print("no input file")
    quit()
# 명령행을 통해 전달된 이미지 경로를 얻어와서 변수에 저장
image_file = sys.argv[1]

# 출력 파일 이름. 정규표현식을 통해 지정된 확장자를 찾은 후 'mosaic'문자열을 변경한다.
output_file = re.sub(r'\.jpg|jpeg|PNG$', '-mosaic1.jpg', image_file)
print("output", output_file)

# 모자이크 강도. 숫자가 클수록 강한 모자이크 효과가 적용됨
mosaic_rate = 30
# 캐스캐이드 파일 경로 지정
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
# 이미지를 Numpy 배열로 변환
image = cv2.imread(image_file)
# 그레이 스케일
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 얼굴 인식 실행하기
cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs,
                                     scaleFactor=1.1,
                                     minNeighbors=1,
                                     minSize=(100,100))
# 얼굴이 감지되지 않으면 프로그램 종료
if len(face_list) == 0:
    print("얼굴을 인식할 수 없습니다. ")
    quit()
# 얼굴이 감지되면 좌상단의 좌표와 가로, 세로 길이를 리스트로 반환
print(face_list)

# 확인한 부분에 모자이크로 처리
color = (0, 0, 255)
for (x,y,w,h) in face_list:
    face_img = image[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (w //mosaic_rate, h // mosaic_rate))
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
    image[y:y + h, x:x + w] = face_img

cv2.imwrite(output_file.replace("resData", "saveFiles"), image)