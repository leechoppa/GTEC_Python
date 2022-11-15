# 하나는 내 모습을 보여주는 영상, 다른 하나는 움직임을 감지하는 흑백 영상 출력
# 영상 거울모드!! np.fliplr()
# 동영상 삽입 -> 경로
# (추가) 텍스트 파일 만들어서 안에 문자 츨력 (=log처리)
import cv2
import numpy as np
import logging

# 움직임 정의하는 픽셀 차이 값.
thresh = 25
max_diff = 10
FRAME_WIDTH = 500
FRAME_HEIGTH = 350

# 프레임 3개
a, b, c = None, None, None

# 카메라 출력
cap = cv2.VideoCapture("Image/sample.mp4")       #()안에 비디오 경로 입력.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)      # 프레임 폭 600
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGTH)     # 프레임 높이 400

# Define the codec and create VideoWriter object. The output is stored in 'output.mp4' file.
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (FRAME_WIDTH, FRAME_HEIGTH))

# 정상적으로 초기화 됐는지 확인  a와 b, 그리고 b와 c
if cap.isOpened():
    ret, a = cap.read()     # a 프레임 읽기
    ret, b = cap.read()     # b 프레임 읽기
    while ret:
        ret, c = cap.read()     # c프레임 읽기
        c = cv2.flip(c,1)       # 카메라 좌우 반전
        draw = c.copy()         # 출력 영상에 사용할 복제본
        
        edge = cv2.Canny(c, 50, 150) # 윤곽선           #c가 맞나? draw인가?

        # 윤곽선은 그레이스케일 영상이므로 저장이 안된다. 컬러 영상으로 변경
        edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        out.write(edge_color) # 영상 데이터만 저장. 소리는 X

        if not ret:
            break
        
        # 3개의 영상을 그레이 스케일로 변경 
        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        
        # a-b, b-c 절대 값 차 구하기
        diff1 = cv2.absdiff(a_gray, b_gray)
        diff2 = cv2.absdiff(b_gray, c_gray)

        # 스레시홀드로 기준치 이내의 차이는 무시
        ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
        ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)

        # 두 차이에 대해서 AND 연산, 두 영상의 차이가 모두 발견된 경우
        diff = cv2.bitwise_and(diff1_t, diff2_t)
        
        # 열림 연산으로 노이즈 제거
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)
        
        # 차이가 발생한 픽셀이 갯수 판단 후 사각형 그리기
        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff:
            nzero = np.nonzero(diff)            # 0이 아닌 픽셀의 좌표 얻기(y[...], x[...])
            cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])), (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)
            logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',       #로그에 시간 출력
                    datefmt ='%m/%d %I:%M:%S %p',   
                    level=logging.DEBUG)
            logging.warning(" 움직임 감지 !")      #로그 출력

            cv2.putText(draw, "Motion Detected!!", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255)) 
            
        stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('motion', stacked)

        # 다음 비교를 위해 영상 순서 정리    
        a = b
        b = c

        # ESC 입력시 프로그램 종료 27(ESC), 13(ENTER), 9(TAB)    
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()