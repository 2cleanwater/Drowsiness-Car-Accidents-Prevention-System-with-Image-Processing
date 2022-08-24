import cv2
import dlib
import opdef
import numpy as np

video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    
def rotate2 (brx,bry,midx,midy,angle):
    
    
    arx = np.cos(-angle)*(brx - midx) - np.sin(-angle)*(bry - midy) 
    ary = np.sin(-angle)*(brx - midx) + np.cos(-angle)*(bry - midy)
    rx = int (arx + midx)
    ry = int (ary + midy)
    
    return(rx,ry)
    
while True:
    ret, frame = video_capture.read()
    orgin_vid = cv2.flip(frame,1)
    clached, detect = opdef.read_video(orgin_vid)
    

    key = cv2.waitKey(10) & 0xFF
    for d in detect:
        shape = shape_predictor(clached,d)
        opdef.landmark_line(orgin_vid,shape,0)
        EAR_Right, EAR_Left = opdef.eye_ratio(shape)
        
           
        
        x = d.left()
        y = d.top()
        x1 = d.right()  
        y1 = d.bottom()
        
        midx = (x+x1)/2
        midy = (y+y1)/2
            
  
        bx = x-(x1-x)/2
        by = y-(y1-y)/2
        bx1 = x1+(x1-x)/2
        by1 = y1+(y1-y)/2
        
        # 큰 d값 저장

        degree = opdef.angle(shape)
            
        cv2.rectangle(orgin_vid, (x, y), (x1, y1), (0, 255, 255), 1)
        opdef.rect_line(orgin_vid,bx,by,bx1,by1)
       
        rx,ry,rx1,ry1 = opdef.rotate(bx,by,bx1,by1,midx,midy,degree)
        rx2,ry2 = rotate2(bx,by,midx,midy,degree)
        rxx = tuple(rx)
        cv2.circle(orgin_vid, rxx, 1, (0,0,255), thickness=5)
        cv2.circle(orgin_vid, (rx2,ry2), 1, (255,255,255), thickness=5)
        print("sibal")
        print(rx,ry,rx1,ry1)
        print(degree)
    
        
    cv2.putText(orgin_vid, "Left : {:.2f}".format(EAR_Left), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)  
    cv2.putText(orgin_vid, "Right : {:.2f}".format(EAR_Right), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
    cv2.putText(orgin_vid, "Degree : {:.2f}".format(degree), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
    cv2.imshow("Frame", orgin_vid)
    
    if key == ord("q"):
        break
     
cv2.destroyAllWindows()
video_capture.release()
