import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import imutils

img1 = cv2.imread('Temp_capture.jpg')
img2 = cv2.imread('doyeop3344.jpg')



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
    print("shape : ",(img.shape))
    img_size = np.asarray(img.shape)[0:2] #np.array는 카피를 생성하고 np.asarray는 레퍼런스을 생성한다.
    print("img_size : " , img_size)
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor) #detect_face.detect_face를 분석하는게 중요하네요.
    
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

def prewhiten(x): #전처리로 명암을 높인다.
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

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
    return -1




distance,prewhitened1, prewhitened2 = compare2face(img1, img2)
threshold = 1.10    # set yourself to meet your requirement 알아서 수정하세요.
print("distance = "+str(distance))
print("Result = " + ("same person" if distance <= threshold else "not same person"))

img1 = imutils.resize(img1,width=1000)
img2 = imutils.resize(img2,width=1000)
pnet = tf.reset_default_graph()
rnet = tf.reset_default_graph()
onet = tf.reset_default_graph()#cv2.imshow를 행하기 전에 뉴럴 네트워크들을 리셋시켜주어야 한다.
cv2.imshow('im1',img1)
cv2.imshow('im2',img2)
cv2.imshow('white1',prewhitened1) #전처리1
cv2.imshow('white2',prewhitened2) #전처리2

k = cv2.waitKey(0) & 0xff
if k==27:
    cv2.destroyAllWindows()



