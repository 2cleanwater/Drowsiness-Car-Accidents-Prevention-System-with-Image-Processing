import cv2
import dlib
import numpy as np
import winsound
import time
import opdef
import sys
import os
import socket

#Server Setting

host = ''
port = 11000 # 포트 11000번으로 설정

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))

sock.listen(1)
conn, addr = sock.accept()


##
video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#비디오를 불러옴, dlib에서 frontal face detector, shape predictor를 불러옴

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
first = 0
#최초 실행 확인여
eye_open_done = True
# 측정을 확인
program_switch = False
#프로그램 트리거
message_popup = 0
#메시지 트리거
print_counter = 0
txt_switch = False
txt_switch2 = False
alarm = False
Login_interface = False

face_alarm = False
face_reco = False
#알람메시지 트리거
#Else
face_reco = False
face_reco_n = True
face = 0
fnd_count = 0
matchConf = None
open_eye = True
identImage = None
Tempimage = None
#로그인 모드 제어 트리거
picTrig = 0
EAR_left = 0
EAR_right = 0

#====================================================================================================================================
    
while True:
     #동영상 좌우반전 후 gray화 및 clahe 후 face detector
    ret, frame = video_capture.read()
    orgin_vid = cv2.flip(frame,1)
    clached, detect = opdef.read_video(orgin_vid)
    
   
    # 키 입력
    key = cv2.waitKey(10) & 0xFF
    
    # 메세지 팝업 결정 
    if message_popup == 1:
        if print_counter == 0:    
            cv2.putText(orgin_vid, "", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 1:    
            cv2.putText(orgin_vid, "Try again", (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 2:
            cv2.putText(orgin_vid, "Gaze the camera", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 3:
            cv2.putText(orgin_vid, "Program starts in : 3", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 4:
            cv2.putText(orgin_vid, "Program starts in : 2", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 5:
            cv2.putText(orgin_vid, "Program starts in : 1", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 6:
            cv2.putText(orgin_vid, "CALCULATING", (240, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print('picTrig : ',picTrig)
    print('Login_interface : ',Login_interface)
    print('program switch: ',program_switch)
    print('match_confirm: ',matchConf)
    print('identImage : ',identImage)
    #Login Mode에 따라서(Follow picTrig) 동작 수행
    #picTrig == 1 이면 새 유저 가입 모드
    if picTrig == 1 and Login_interface == True:
        cv2.putText(orgin_vid, "Look at the cam and Press 'h'", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if key == ord('h'):
            cv2.imwrite("Temp_capture.jpg",orgin_vid)
            Temp_name = input("Put your name in : ")
            path = "./"
            newim = None
            for filename in os.listdir(path):
                if filename.startswith('Temp_capture'):
                    os.rename(filename,Temp_name+'.jpg')
                    newim = Temp_name+'.jpg'
                    
            print('newim:', newim)
            #새 이미지를 불러들임
            
            earsaveim = cv2.imread(newim)
            Tempimage = cv2.imread(newim)
            cv2.imshow('Lets See: ',Tempimage)
            print('earsaveim : ',earsaveim)
            #새 이미지의 clahe 이미지, rectangle 좌표 획득
            
            earsave_clahe,earsave_detection = opdef.read_video(earsaveim)
            print('earsave_detection type : ',type(earsave_detection))
            print('earsave_detection : ',earsave_detection)
            #Face landmark 좌표
            for fd in earsave_detection:
                #shape_predictor가 shape 좌표 반환
                nushape = shape_predictor(earsave_clahe,fd)
                opdef.landmark_line(earsaveim,nushape,0)

            print('earsaveim22:',type(earsaveim))
            if nushape is None:
                print("Error : Sorry, We couldn't find your face.")
                cv2.destroyAllWindows()
                sys.exit()
            
            saveear = opdef.eye_ratio(nushape)
            
            # 파일명이 담긴 배열 요소를 이름으로 파일을 저장하는 과정
            np.save('{}'.format(newim),saveear) 

            ##}##동작 Part
            #이미지 저장 후 프로그램 초기 상태로
            Login_interface = True
            picTrig = 0 
            
            
    #picTrig ==2 이면 로그인 모드
    elif picTrig == 2 and Login_interface == True:
        cv2.putText(orgin_vid, "Look at the cam and Press 'h'", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if key == ord('h'):
            cv2.imwrite("Temp_capture.jpg",orgin_vid)
            matchConf,identImage = opdef.facematching("Temp_capture.jpg")
            Login_interface = False
            picTrig = 0
        elif matchConf == False:
            cv2.putText(orgin_vid, "Verifying Failed. Sign up Please. Press 'P' ", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if matchConf == True and Login_interface ==False and picTrig ==0:
        EARarr = np.load('{}.npy'.format(identImage))
        eye_ratio_limit = (EARarr[0]+EARarr[1]) /2*0.65
        program_switch = True  
        print('loaded and got limit array : ',eye_ratio_limit)
    
    if key == ord("p"):
        #p 눌렸을 때 인식 측정 시작 
        if not eye_cap == True:
            Login_interface = True
            
        
        #다시 누르면 이미지창 제거 메세지창 제거
        else :
            eye_open_done = True
            eye_cap = False
            cv2.destroyWindow("image2")
            program_switch = False
            eye_ratio_limit = 0.00
            left_ar = [0,]
            right_ar = [0,]
            Login_interface = True
            count_ar = 0
            winsound.PlaySound(None, winsound.SND_ASYNC)
            txt_switch = False
            alarm = False
            
         ##picTrig == 0 and Login_interface == True이면 초기화
    if picTrig ==0 and Login_interface == True:
        cv2.putText(orgin_vid, "New Register? Press N", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(orgin_vid, "Or Sign In? Press I", (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        #n을 누르면 새 사용자 등록
        #i를 누르면 기존 사용자 로그인
        if key == ord('n'):
            print('New Register')
            picTrig = 1
            print(Login_interface, picTrig)
            
        elif key == ord('i'):
            print('Sign in')
            picTrig = 2
            print(Login_interface, picTrig)
            
    
                
#================================================================================================================================
        
    if program_switch == True:
        #프로그램 시작
        face_reco = False
        face_reco_n = True
        
        for d in detect:
            
            face_reco = True
            fnd_count = 0
            count_time2 = time.time()
            
            if txt_switch2 == True:
                winsound.PlaySound(None, winsound.SND_ASYNC)
                face_alarm = False    
            txt_switch2 = False
            #얼굴 인식 불가 알람이 ON일때 알람을 끔
            
            shape = shape_predictor(clached, d)
            
            x = d.left()
            y = d.top()
            x1 = d.right()  
            y1 = d.bottom()
            #d 값 저장
            
            bx = x-(x1-x)/2
            by = y-(y1-y)/2
            bx1 = x1+(x1-x)/2
            by1 = y1+(y1-y)/2
            
            # 큰 d값 저장

            degree,angle = opdef.angle(shape)
            #좌표 회전
            
            rx,ry,rx1,ry1 = opdef.rotate(x,y,x1,y1,angle)
            rbx,rby,rbx1,rby1 = opdef.rotate(bx,by,bx1,by1,angle)
            

            pts1 = np.array([rbx,rby,rbx1,rby1],np.float32)
            pts2 = np.array([[0,0],[400,0],[0,400],[400,400]],np.float32)
    
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(orgin_vid,M,(400,400))
            
            #회전된 좌표를 이용하여 새로운 창으로 프린트
            
            
 #=======================================================================================================================
            # 회전된 d2에서 얼굴 인식 실행

            d2clahe_image, d2detections = opdef.read_video(dst)
    
            for d2 in d2detections:
                              
                xx = d2.left()  
                yy = d2.top()  
                xx1 = d2.right()  
                yy1 = d2.bottom()
                
                d2shape = shape_predictor(d2clahe_image, d2)
                
                cv2.rectangle(dst, (xx, yy), (xx1, yy1), (0, 255, 255), 1)
                opdef.landmark_line(dst,d2shape,0)
                EAR_right, EAR_left = opdef.eye_ratio(d2shape)
                print(EAR_right)
                    
                if EAR_left <= eye_ratio_limit and EAR_right <= eye_ratio_limit:
                    open_eye = False
                    sendlen = conn.send(bytes("False","utf-8"))
                    if(sendlen == None or sendlen == ''):
        
                        print('SendError')
                        break 
                    else: 
                        #성공적으로 보냈을 시 True라고 출력
                        print('Sended Flag : ',True)
                if EAR_left > eye_ratio_limit and EAR_right > eye_ratio_limit:
                    open_eye = True
                    sendlen = conn.send(bytes("True","utf-8"))
                    if(sendlen == None or sendlen == ''):
        
                        print('SendError')
                        break 
                    else: 
                        #성공적으로 보냈을 시 True라고 출력
                        print('Sended Flag : ',True)
                # 눈 감았을때 open eye가 꺼짐, 눈 뜨면 다시 켜짐
                
                if open_eye == True:
                    count_time = time.time()
                
                #눈이 감겨있으면 그 순간 count_time 측정 기록, 떠있으면 계속 갱신
            
        
            opdef.rect_line(orgin_vid,rx,ry,rx1,ry1)
            
            #회전된 작은 d 프린트
            
            opdef.landmark_line(orgin_vid,shape,0)
             
            cv2.putText(orgin_vid, "Left : {:.2f}".format(EAR_left), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)  
            cv2.putText(orgin_vid, "Right : {:.2f}".format(EAR_right), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(orgin_vid, "Degree : {:.2f}".format(degree), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(orgin_vid, "Eye ratio limit: {:.2f}".format(eye_ratio_limit), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
                cv2.putText(orgin_vid, "FACE NOT DETECTED", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            count_time = time.time()
            
            if txt_switch == False and time.time()-count_time2 > 4.5:
                txt_switch2 = True
                if txt_switch2 == True:
                    cv2.putText(orgin_vid, "NO FACE ALARM!!!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
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
            cv2.putText(orgin_vid, "ALARM!!!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
        #눈 감으면 알람 
        
        cv2.imshow("image2", dst)
        
        #회전된 이미지 창 출력
        
        if face == 2 and key == ord("s"):
            
            txt_switch = False
            alarm = False
            winsound.PlaySound(None, winsound.SND_ASYNC)
                    
        #알람 울릴 때  (얼굴 인식이 되는 경우에만 알람 종료)
        
    cv2.imshow("Frame", orgin_vid)
    
    #print("time = ",time.time()-count_time)
    #print("time2 = ",time.time()-count_time2)
    #print("face = ",face)
    
    if key == ord("q"):
        
        eye_ratio_limit = 0.00
        winsound.PlaySound(None, winsound.SND_ASYNC)
        break
    # 아예 종료
        
cv2.destroyAllWindows()
video_capture.release()
