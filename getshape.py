import cv2
import dlib
import numpy as np
import sys
import opdef
import Myfacematch_Temp
 

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
nushape = None
lists = Myfacematch_Temp.imgfiles('./')
print(lists)
#이미지 선
newim = Myfacematch_Temp.File_filtering(lists,Myfacematch_Temp.User_Authentication())

print('newim : ',newim)
print('newim : ', newim[0])
print('newim : ', type(newim[0]))
path = "./"
    #새 이미지를 불러들임
earsaveim = cv2.imread(newim[0])
cv2.imshow('Before',earsaveim)
    #새 이미지의 clahe 이미지, rectangle 좌표 획득    
earsave_clahe,earsave_detection = opdef.read_video(earsaveim)
print('earsave_detection type : ',type(earsave_detection))

cv2.imshow('earsave_clahe',earsave_clahe)

#Face landmark 좌표 획득

#detection된 변수(dlib.rectangle Type)을 for문을 통해 돌려서 
#shape_predictor에 인자를 전달하여 nushape를 구성해야 한다.
for fd in earsave_detection:
    #shape_predictor가 shape 좌표 반환
    nushape = shape_predictor(earsave_clahe,fd)
    opdef.landmark_line(earsaveim,nushape,0)

print('earsaveim',type(earsaveim))

#랜드마크 적용 및 좌표 그려놓은 이미지 출력
cv2.imshow('after ',earsaveim)
#Ear값 저장
print('nushape : ',nushape)
print('nushape type : ',type(nushape))
# Face Landmark가 생성되지 못한 경우 프로그램 종료
if nushape is None:
        print("Error : Sorry, We couldn't find your face.")
        cv2.destroyAllWindows()
        sys.exit()
saveear = opdef.eye_ratio(nushape) #nushape에서 EAR 획득

print('svear : ',saveear)
##print('Lets fuck this out : ',newim[0])

# 파일명이 담긴 배열 요소를 이름으로 파일을 저장하는 과정
np.save('{}'.format(newim[0]),saveear) 

# 저장한 리스트 객체를 불러오는 코드
loaddata = np.load('{}.npy'.format(newim[0]))
print('loaddata : ',loaddata)
##print('loaddata : ',loaddata)
key = cv2.waitKey(0) & 0xFF
if key == ord('q'):
    cv2.destroyAllWindows()