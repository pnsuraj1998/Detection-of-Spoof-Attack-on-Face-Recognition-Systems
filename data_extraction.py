import argparse
import pandas as pd
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
        for file in os.listdir(path):
            img=cv2.imread(os.path.join(path,file))
            face=extract_face(img)
            patches=extract_patches(face)
            
            for i in range(len(patches)):
                if file[0]=='f':
                    data.append((patches[i],0))
                else:
                    data.append((patches[i],1))
            
            df=pd.DataFrame(data)
            df.to_csv("data.csv",index=None,header=None)
        
        

        




