import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import winsound
import time


video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#비디오를 불러옴, dlib에서 frontal face detector, shape predictor를 불러옴


right_eye_points = list(range(36, 42))
left_eye_points = list(range(42, 48))
#오른쪽 왼쪽 눈의 shape p 상 좌표를 어레이로 저장

count_ar = 0
left_ar = 30*[]
right_ar = 30*[]
#count_ar 가 증가하는 동안 left,right_ar에 30개씩 ear값 저장


eye_ratio_limit = 0.00
#ear 값을 0으로 설정,

count_time = 0
count_time2 = 0


eye_cap = False
#p를 눌러서 측정이 되었는지 확인하는 트리거
eye_open_done = True
# 측정을 확인
program_switch = False
#프로그램 트리거
message_popup = False
#메시지 트리거
print_counter = 0
txt_switch = False
txt_switch2 = False
alarm = False

face_alarm = False
face_reco = False
#알람메시지 트리거

face_reco = False
face_reco_n = True
face = 0
fnd_count = 0

open_eye = True

def eye_ratio(eyepoint):
    A = dist.euclidean(eyepoint[1],eyepoint[5])
    B = dist.euclidean(eyepoint[2],eyepoint[4])
    C = dist.euclidean(eyepoint[0],eyepoint[3])
    EAR = (A+B) / (2.0*C)
    
    return EAR

#EAR 값 계산 define

def rotate (brx,bry):
    crx = brx - midx
    cry = bry - midy
    arx = np.cos(-angle)*crx - np.sin(-angle)*cry 
    ary = np.sin(-angle)*crx + np.cos(-angle)*cry
    rx = int (arx + midx)
    ry = int (ary + midy)
    
    return(rx,ry)

#점 회전 define    

#====================================================================================================================================
    
while True:
    ret, frame = video_capture.read()
    flip_frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detection = face_detector(clahe_image)
    
    #동영상 좌우반전 후 gray화 및 clahe 후 face detector
    key = cv2.waitKey(10) & 0xFF
    # 키 입력
    
    if message_popup == True:
        if print_counter == 0:    
            cv2.putText(flip_frame, "", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 1:    
            cv2.putText(flip_frame, "Try again", (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 2:
            cv2.putText(flip_frame, "Gaze the camera", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 3:
            cv2.putText(flip_frame, "Program starts in : 3", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 4:
            cv2.putText(flip_frame, "Program starts in : 2", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 5:
            cv2.putText(flip_frame, "Program starts in : 1", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 6:
            cv2.putText(flip_frame, "CALCULATING", (240, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 메세지 팝업 결정 
    
    if key == ord("p"):
        if not eye_cap == True:
            
            eye_open_done = False
            
        #p 눌렸을 때 인식 측정 시작 
        else :
            eye_open_done = True
            eye_cap = False
            cv2.destroyWindow("image2")
            program_switch = False
            eye_ratio_limit = 0.00
            left_ar = [0,]
            right_ar = [0,]
            
            count_ar = 0
            winsound.PlaySound(None, winsound.SND_ASYNC)
            txt_switch = False
            alarm = False
            
        #다시 누르면 이미지창 제거 메세지창 제거 

    if eye_open_done == False:
        # 측정 시작
        for fd in detection:
            eye_open_shape = shape_predictor(clahe_image, fd)
            
            eye_open_landmarks = np.matrix([[p.x, p.y] for p in eye_open_shape.parts()])

            eye_open_left_eye = eye_open_landmarks[left_eye_points]
            eye_open_right_eye = eye_open_landmarks[right_eye_points]

            eye_open_EAR_left = eye_ratio(eye_open_left_eye)
            eye_open_EAR_right = eye_ratio(eye_open_right_eye)
            
            # EAR값 측정 시작
            if(count_ar < 100):
                count_ar += 1
                
                for i in range(36,41):
                    cv2.line(flip_frame,(eye_open_shape.part(i).x, eye_open_shape.part(i).y),(eye_open_shape.part(i+1).x, eye_open_shape.part(i+1).y),(255,0,0),1)
                    cv2.line(flip_frame,(eye_open_shape.part(41).x, eye_open_shape.part(41).y),(eye_open_shape.part(36).x, eye_open_shape.part(36).y),(255,0,0),1)
                for i in range(42,47):
                    cv2.line(flip_frame,(eye_open_shape.part(i).x, eye_open_shape.part(i).y),(eye_open_shape.part(i+1).x, eye_open_shape.part(i+1).y),(255,0,0),1)
                cv2.line(flip_frame,(eye_open_shape.part(47).x, eye_open_shape.part(47).y),(eye_open_shape.part(42).x, eye_open_shape.part(42).y),(255,0,0),1)
                print_counter = 2
                message_popup = True
                
                if(30<count_ar<=60):
                    left_ar.append(eye_open_EAR_left)
                    right_ar.append(eye_open_EAR_right)
                    print_counter = 6
                if(60<count_ar<=70):
                    print_counter = 0
                    Max_EAR_left = max(left_ar)
                    Max_EAR_right = max(right_ar)
                    eye_ratio_limit = (Max_EAR_left + Max_EAR_right)/2*0.65
                if(70<count_ar<=80):
                    print_counter = 3        
                if(80<count_ar<=90):
                    print_counter = 4
                if(90<count_ar<100):
                    print_counter = 5
                    

            #얼굴이 인식되는 동안 count_ar이 올라가면서 어레이에 저장후 최대값으로 설정, 메시지 팝업

        

        if(count_ar == 100):
            eye_open_done = True
            eye_cap = True       
            program_switch = True
            print_counter = 0
            count_ar = 0
            count_time = time.time()
        
        #count_ar이 최대일떄 측정 중단, 프로그램 시작
                
#================================================================================================================================
        
    if program_switch == True:
        #프로그램 시작
        face_reco = False
        face_reco_n = True
        
        for d in detection:
            
            face_reco = True
            fnd_count = 0
            count_time2 = time.time()
            
            if txt_switch2 == True:
                    winsound.PlaySound(None, winsound.SND_ASYNC)
                    face_alarm = False    
            txt_switch2 = False
            #얼굴 인식 불가 알람이 ON일때 알람을 끔
            
            x = d.left()
            y = d.top()
            x1 = d.right()  
            y1 = d.bottom()
            #d 값 저장
            bdx = x-(x1-x)/2
            bdy = y-(y1-y)/2
            bdx1 = x1+(x1-x)/2
            bdy1 = y1+(y1-y)/2
            # 큰 d값 저장
            midx = (x+x1)/2
            midy = (y+y1)/2
            # d의 가운데 포인트 저장
        
            shape = shape_predictor(clahe_image, d)
            
            rex = shape.part(45).x
            rey = shape.part(45).y
            lex = shape.part(36).x
            ley = shape.part(36).y
            
            mex = int (lex + (rex-lex)/2)
            mey = int (ley + (rey-ley)/2)
            #눈의 양끝점 좌표 설정 및 눈 사이 가운데 점 설정
            
            tanx = mex - lex
            tany = ley - mey
            tan = tany/tanx
            #tan 값 계산
            angle = np.arctan(tan)
            degree = np.degrees(angle)
            #각도 계산
            
            rsd_1 = rotate(x,y)
            rsd_2 = rotate(x1,y)
            rsd_3 = rotate(x,y1)
            rsd_4 = rotate(x1,y1)
            d2_1 = rotate(bdx,bdy)
            d2_2 = rotate(bdx1,bdy)
            d2_3 = rotate(bdx,bdy1)
            d2_4 = rotate(bdx1,bdy1)

            
            #좌표 회전
    
            
            pts1 = np.float32([[d2_1[0],d2_1[1]],[d2_2[0],d2_2[1]],[d2_3[0],d2_3[1]],[d2_4[0],d2_4[1]]])
            pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
    
            M = cv2.getPerspectiveTransform(pts1,pts2)
    
            dst = cv2.warpPerspective(flip_frame,M,(400,400))
            
            #회전된 좌표를 이용하여 새로운 창으로 프린트
            
            
 #=======================================================================================================================
            # 회전된 d2에서 얼굴 인식 실행
            d2gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            d2clahe_image = clahe.apply(d2gray)
            d2detections = face_detector(d2clahe_image)
            
    
            for d2 in d2detections:
                
                
                xx = d2.left()  
                yy = d2.top()  
                xx1 = d2.right()  
                yy1 = d2.bottom()
                d2shape = shape_predictor(d2clahe_image, d2)
                
                cv2.rectangle(dst, (xx, yy), (xx1, yy1), (0, 255, 255), 1)
                for i in range(1,68):
                    cv2.circle(dst, (d2shape.part(i).x, d2shape.part(i).y), 1, (255,0,255), thickness=1) 
                for i in range(36,41):
                    cv2.line(dst,(d2shape.part(i).x, d2shape.part(i).y),(d2shape.part(i+1).x, d2shape.part(i+1).y),(255,0,0),1)
                    cv2.line(dst,(d2shape.part(41).x, d2shape.part(41).y),(d2shape.part(36).x, d2shape.part(36).y),(255,0,0),1)
                for i in range(42,47):
                    cv2.line(dst,(d2shape.part(i).x, d2shape.part(i).y),(d2shape.part(i+1).x, d2shape.part(i+1).y),(255,0,0),1)
                cv2.line(dst,(d2shape.part(47).x, d2shape.part(47).y),(d2shape.part(42).x, d2shape.part(42).y),(255,0,0),1)
                
                landmarks = np.matrix([[p.x, p.y] for p in d2shape.parts()])
            
                right_eye = landmarks[left_eye_points]
                left_eye = landmarks[right_eye_points]
        
                EAR_right = eye_ratio(right_eye)            
                EAR_left = eye_ratio(left_eye)
                    
                
                if EAR_left <= eye_ratio_limit and EAR_right <= eye_ratio_limit:
                    open_eye = False
                    
                if EAR_left > eye_ratio_limit and EAR_right > eye_ratio_limit:
                    open_eye = True
                    
                # 눈 감았을때 open eye가 꺼짐, 눈 뜨면 다시 켜짐
                
                if open_eye == True:
                    count_time = time.time()
                
                #눈이 감겨있으면 그 순간 count_time 측정 기록, 떠있으면 계속 갱신
                    
            cv2.line(flip_frame,rsd_1,rsd_2,(100,255,100),1)
            cv2.line(flip_frame,rsd_1,rsd_3,(100,255,100),1)
            cv2.line(flip_frame,rsd_4,rsd_2,(100,255,100),1)
            cv2.line(flip_frame,rsd_4,rsd_3,(100,255,100),1)
            #회전된 작은 d 프린트
        
            for i in range(0,67):
                cv2.circle(flip_frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1)
            for i in range(36,41):
                cv2.line(flip_frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),1)
            cv2.line(flip_frame,(shape.part(41).x, shape.part(41).y),(shape.part(36).x, shape.part(36).y),(255,0,0),1)
            for i in range(42,47):
                cv2.line(flip_frame,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),1)
            cv2.line(flip_frame,(shape.part(47).x, shape.part(47).y),(shape.part(42).x, shape.part(42).y),(255,0,0),1)    
            
            cv2.putText(flip_frame, "Left : {:.2f}".format(EAR_left), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)  
            cv2.putText(flip_frame, "Right : {:.2f}".format(EAR_right), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(flip_frame, "Degree : {:.2f}".format(degree), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(flip_frame, "Eye ratio limit: {:.2f}".format(eye_ratio_limit), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            #라인 그리기 및 값 표현
            
            if time.time()-count_time > 2.5:
                txt_switch = True
                
                if alarm == False:
                    winsound.PlaySound("school_alarm.wav", winsound.SND_LOOP + winsound.SND_ASYNC)
                    print("alarm on")
                alarm = True
            
            #2.5초 후 알람 한번 출력, 메시지 스위치 온    
        
          
        if face_reco == False:
            face_reco_n = False
            fnd_count += 1
            if fnd_count >= 10:
                cv2.putText(dst, "FACE NOT DETECTED", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(flip_frame, "FACE NOT DETECTED", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            count_time = time.time()
            
            if txt_switch == False and time.time()-count_time2 > 4.5:
                txt_switch2 = True
                if txt_switch2 == True:
                    cv2.putText(flip_frame, "NO FACE ALARM!!!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                if face_alarm == False:
                    winsound.PlaySound("school_alarm.wav", winsound.SND_LOOP + winsound.SND_ASYNC)
                    print("face alarm on")
                face_alarm = True
                
        #얼굴 인식이 안될 경우 눈 감는 카운트 초기화, 4.5 초 후 알람 및 메시지 출력
                
                
        if face_reco == True and face_reco_n == True:
            face = 2
            #얼굴 인식이 되는 도중
        if face_reco == False and face_reco_n == False:
            face = 1
            #얼굴 인식이 안되는 중
        if face_reco == False and face_reco_n == True:
            face = 0
            #프로그램 시작 값
            
        if txt_switch == True:
            cv2.putText(flip_frame, "ALARM!!!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
        #눈 감으면 알람 
        
        cv2.imshow("image2", dst)
        
        #회전된 이미지 창 출력
        
        if face == 2 and key == ord("s"):
            
            txt_switch = False
            alarm = False
            winsound.PlaySound(None, winsound.SND_ASYNC)
                    
        #알람 울릴 때  (얼굴 인식이 되는 경우에만 알람 종료)
        
    cv2.imshow("Frame", flip_frame)
    
    print("time = ",time.time()-count_time)
    print("time2 = ",time.time()-count_time2)
    print("face = ",face)
    
    if key == ord("q"):
        
        eye_ratio_limit = 0.00
        winsound.PlaySound(None, winsound.SND_ASYNC)
        break
    # 아예 종료
        
cv2.destroyAllWindows()
video_capture.release()
