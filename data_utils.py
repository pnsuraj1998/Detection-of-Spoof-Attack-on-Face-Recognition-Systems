from facenet_pytorch import MTCNN
import cv2
from PIL import Image

def extract_keypoints(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sift=cv2.xfeatures2d.SIFT_create()
    key_points,_=sift.detectAndCompute(img,None)
    points=[]
    for i in range(0,len(key_points)):
        points.extend([key_points[i][0],key_points[i][1]])
    return points

def extract_patches(img):
    key_points=extract_keypoints(img)
    W,H=img.shape
    patch=[]
    for i in range(len(key_points)):
        x,y=key_points[i][0],key_points[i][1]
        start_x=max(0,int(x-96/2))
        start_y=max(0,int(y-96/2))
        end_x=min(W,int(x+96/2))
        end_y=min(H,int(y+96/2))
        img_patch=img[start_x:end_x,start_y:end_y]
        patch.extend(cv2.resize(img_patch,(96,96)))
    return patch

def extract_face(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img=Image.fromarray(img)
    mtcnn=MTCNN(select_largest=False,selection_method="probability")
    boxes,prob=mtcnn.detect(img,landmarks=False)
    box=boxes[prob.index(max(prob))]
    face=img[box[0]:box[2],box[1]:box[3]]
    return face