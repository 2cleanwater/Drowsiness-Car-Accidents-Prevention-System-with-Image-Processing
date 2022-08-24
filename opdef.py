import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

import Myfacematch_Temp 
from Myfacematch_Temp import tf
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")




#Image의 Clahe화와 변환된 이미지로부터 얼굴을 detection한다.
def read_video(video):
    
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detection = face_detector(clahe_image)
    
    return (clahe_image,detection)
    

def make_clahe(grayimage):
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(grayimage)
    detection = face_detector(clahe_image)
    
    return (clahe_image,detection)
def eye_ratio(shape):
    
    right_eye_points = list(range(36, 42))
    left_eye_points = list(range(42, 48))

    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

    right_eye = landmarks[left_eye_points]
    left_eye = landmarks[right_eye_points]

    A = dist.euclidean(right_eye[1],right_eye[5])
    B = dist.euclidean(right_eye[2],right_eye[4])
    C = dist.euclidean(right_eye[0],right_eye[3])
    D = dist.euclidean(left_eye[1],left_eye[5])
    E = dist.euclidean(left_eye[2],left_eye[4])
    F = dist.euclidean(left_eye[0],left_eye[3])
    
    EAR_R = (A+B) / (2.0*C)
    EAR_L = (D+E) / (2.0*F)
       
    return (EAR_R,EAR_L)

def rotate (x,y,x1,y1,angle):
    
    
    midx = (x+x1)/2
    midy = (y+y1)/2

    points = [x,x1,y,y1]
        
    k=0
    r = [0,0,0,0]
    
    for j in range(2,4):
        for i in range(0,2):
                
                arx = np.cos(-angle)*(points[i] - midx) - np.sin(-angle)*(points[j] - midy)
                ary = np.sin(-angle)*(points[i] - midx) + np.cos(-angle)*(points[j] - midy)
                rx = int (arx + midx)
                ry = int (ary + midy)
                r[k]=[rx,ry]
                k=k+1
            
    return(r[0],r[1],r[2],r[3])
    


def angle (shape):
              
    rex = shape.part(45).x
    rey = shape.part(45).y
    lex = shape.part(36).x
    ley = shape.part(36).y
    
    mex = int (lex + (rex-lex)/2)
    mey = int (ley + (rey-ley)/2) 
    
    tanx = mex - lex
    tany = ley - mey
    tan = tany/tanx

    angle = np.arctan(tan)
    degree = np.degrees(angle)
    
    return (degree,angle)
    
def landmark_line (orgin_vid,shape,a):
    
    if a == 0:
        for i in range(0,67):
            cv2.circle(orgin_vid, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=-1)
    for i in range(36,41):
        cv2.line(orgin_vid,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),1)
    cv2.line(orgin_vid,(shape.part(41).x, shape.part(41).y),(shape.part(36).x, shape.part(36).y),(255,0,0),1)
    for i in range(42,47):
        cv2.line(orgin_vid,(shape.part(i).x, shape.part(i).y),(shape.part(i+1).x, shape.part(i+1).y),(255,0,0),1)
    cv2.line(orgin_vid,(shape.part(47).x, shape.part(47).y),(shape.part(42).x, shape.part(42).y),(255,0,0),1)
                  

def rect_line (orgin_vid,x,y,x1,y1):
    
    x = tuple(x)
    y = tuple(y)
    x1 = tuple(x1)
    y1 = tuple(y1)
    cv2.line(orgin_vid,x,x1,(100,255,100),1)
    cv2.line(orgin_vid,x,y,(100,255,100),1)
    cv2.line(orgin_vid,y,y1,(100,255,100),1)
    cv2.line(orgin_vid,x1,y1,(100,255,100),1)
            
def facematching(anchor):
    Imagelist = Myfacematch_Temp.imgfiles('./')

    print('ImageList : ',Imagelist)
    Filtered = Myfacematch_Temp.File_filtering(Imagelist,Myfacematch_Temp.User_Authentication())
    if not Filtered == []:
        print ("Filtered Files: ", Filtered)
    elif Filtered == []:
        print("There is no such named file...")
        Filtered.append(None)
        pnet = tf.reset_default_graph()
        rnet = tf.reset_default_graph()
        onet = tf.reset_default_graph()
        print('pnet : ',pnet)
        print('rnet : ',rnet)
        print('onet : ',onet)
        print('They are Intialized')
    temp = None
    match_bool = []
    # temp 사진 삽입.
    temp = anchor

#images 변수는 String. 이미지 파일의 이름을 뜻한다.

    for images in Filtered:
        print('imagesRecog : ',images) # 알아볼수 있게 앞에 문자열 추가해보고 디버깅하기 18_04_10 20:05 End
        match_bool = Myfacematch_Temp.face_matching(temp,images)

#최종 확인 여부
    print("Confirmed? : ",match_bool)
    return match_bool ,Filtered[0]



    