import argparse
import h5py
import os
from data_utils import *

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
	                help="path to Dataset")
    args = vars(ap.parse_args())
    
    if not args.get("input"):
        raise ValueError("Path to Dataset should not be empty")
    else:
        ''' Assuming that there is no other folders in the dataset folder. All the samples are present together in the same folder.
            Each sample is labelled with a name starting with f (indicating spoof image) and r (indicating real image)'''
        path=args.get("input")
        data=[]
        labels=[]
        for file in os.listdir(path):
            img=cv2.imread(os.path.join(path,file))
            face=extract_face(img)
            patches=extract_patches(face)
            
            for i in range(len(patches)):
                if file[0]=='f':
                    data.append(patches[i])
                    labels.append([0])
                else:
                    data.append(patches[i])
                    labels.append([1])
            
            hf = h5py.File('data.h5', 'w')
            train_features=hf.create_dataset('Dataset_Data', data=data)
            train_labels=hf.create_dataset('Dataset_Labels', data=labels)
            hf.close()
