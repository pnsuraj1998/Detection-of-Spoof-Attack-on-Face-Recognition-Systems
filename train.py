from model import patch_cnn_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import *
from torch.utils.data import DataLoader
import torch.optim as optim

def train(model,loss_fn,train_loader):
    params=model.params()
    optimizer=optim.Adam(params,lr=0.001, betas=(0.9, 0.999), eps=1e-08)





if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
	                help="path to h5 File")
    args = vars(ap.parse_args())

    if not args.get("input"):
        raise argparse.ArgumentError("h5 file path isn't specified")
    else:
        X,Y= retrieve_data(args.get("input"))
        n=len(X)
        dataset=create_Dataset(X,Y,transform=None)
        train_dataset,test_dataset=torch.utils.data.random_split(dataset,(int(n*0.8),int((n*0.2))))
        train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
        test_loader=DataLoader(test_dataset,batch_size=1)

        model=patch_cnn_model()
        loss_fn=F.cross_entropy()

        epochs=1000






