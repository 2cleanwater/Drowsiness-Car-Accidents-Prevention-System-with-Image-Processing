#Myfacenet_Temp.py

import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import imutils
import os
#tf.logging.set_verbosity(tf.logging.ERROR)

#for root, dirs, files in os.walk('C:/Users/KimDoYeop/Documents/Pyproject/facematch-master'):
#    for file in files:
#        print (file)
#os.walk -> 디렉토리를 순회하면서 파일을 반환.
#os.remove(path) -> 파일 삭제. path가 디렉토리면 OSError 발생.

#facematching Project.
#Input : Userface_Temp , images in directory(filtered)
#Output : match(bool type value.)


#현재 디렉토리 내에서 Image파일명만 추출하여 리스트로 반환
def imgfiles(path):
    res = []
    for path, dirs, files in os.walk(path): #모든 파일을 walk하며 관련 값 반환        
        for file in files:        
            if '.jpg' in os. path.splitext(file):
                res.append(file)        
    return res
#def Call(img1)
def File_filtering(Imlists,name):
    Arr = []
    for Imfile in Imlists:
        if name in Imfile:
            print(name,"을 배열에 추가")
            Arr.append(Imfile)
    return Arr
                
def User_Authentication():
    u_name = input("당신의 이름을 입력하세요. : ")
    
    return u_name



def face_matching(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if (type(img1[0]) !=type( None)) and  (type(img2[0]) != type(None)) :
        print("face_matching Func has right images!!")
    else :
        print("face_matching Func must have right images!!")
        return False
    # some constants kept as default from facenet
    minsize = 20    
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    margin = 44
    input_image_size = 160
    
    sess = tf.Session()
    
    # read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
    pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')
    
    # read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
    facenet.load_model("20170512-110547/20170512-110547.pb")
    
    # Get input and output tensors # 입출력 텐서들을 받습니다.
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    print(embeddings)
    embedding_size = embeddings.get_shape()[1]
    
    def getFace(img): #얼굴 검출 함수 선언 detect_face.detect_face에서 얻은 얼굴 위치를 적절히 잘라 다듬고 대비 극대화 전처리 후 faces 배열로 반환합니다.
        faces = []
        if type(img) == type(None):
            return None, -1
        print("shape : ",(img.shape))
        img_size = np.asarray(img.shape)[0:2] #np.array는 카피를 생성하고 np.asarray는 레퍼런스을 생성한다.
        print("img_size : " , img_size)
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor) #detect_face.detect_face를 분석하는게 중요하네요.
        
        if len(bounding_boxes) == 0:
            print("I couldn't find your face. Take again Please.")
            return None, -1
        
        print("bounding box Var: ", bounding_boxes)
        
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                if face[4] > 0.50: #face[4]가 0.5 이상인 경우에,
                    det = np.squeeze(face[0:4])
                    print("det : ", det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :] #이미지다듬기
                    resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                    prewhitened = facenet.prewhiten(resized)
                    faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
        return faces, prewhitened 
    
    def getEmbedding(resized):
        reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
        feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
        embedding = sess.run(embeddings, feed_dict=feed_dict)
        return embedding
    
    def compare2face(img1,img2):
        face1, prewhitened1 = getFace(img1)
        face2, prewhitened2 = getFace(img2)
        if face1 and face2:
            # calculate Euclidean distance
            dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
            return dist , prewhitened1,prewhitened2
        return 100,-1,-1
        
    distance,prewhitened1, prewhitened2 = compare2face(img1, img2)
    threshold = 1.10    # set yourself to meet your requirement 알아서 수정하세요.
    
    if distance<= threshold:
        match = True
    else :
        match = False
    
    print("distance = "+str(distance))
    print("Result = " + ("same person" if distance <= threshold else "not same person"))
    
    img1 = imutils.resize(img1,width=480)
    img2 = imutils.resize(img2,width=480)
    pnet = tf.reset_default_graph()
    rnet = tf.reset_default_graph()
    onet = tf.reset_default_graph()#cv2.imshow를 행하기 전에 뉴럴 네트워크들을 리셋시켜주어야 한다.
    cv2.imshow('im1',img1)
    cv2.imshow('im2',img2)
    cv2.imshow('white1',prewhitened1) #전처리1
    cv2.imshow('white2',prewhitened2) #전처리2
    
    return match
    #Path : C:/Users/user/Desktop/MyFacematch 연구실 컴
    #Path : C:/Users/KimDoYeop/Documents/Pyproject/facematch-master 노트북
    




