import argparse
import torch
import os
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

def infer(model,patches):
    model.eval()
    result=[0,0]
    with torch.no_grad():
        for i in range(len(patches)):
            output=model(patches[i])
            result[torch.argmax(output)]+=1
    return result.index(max(result))

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
	                help="path to Model")
    args = vars(ap.parse_args())

    if not args.get("input"):
        raise argparse.ArgumentError("Path to Model should not be empty")
    PATH=args.get("input")
    if not os.path.exists(PATH):
        raise ValueError("There is no model at specified path")
    cap=cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret,frame=cap.read()
        if ret!=True:
            break
        face=extract_face(frame)
        points=extract_keypoints(face)
        patches=extract_patches(points)

        model=torch.load(args.get("input"))
        final_result=infer(model,patches)
        if final_result==1:
            cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0,255,0), 1)
            cv2.putText(frame, 'Live_Face', (face[0], face[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0,0,255), 1)
            cv2.putText(frame, 'Spoof_Face', (face[0], face[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        cv2.imshow('Spoof Detection',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

              





