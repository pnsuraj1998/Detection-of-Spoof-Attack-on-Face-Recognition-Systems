from model import patch_cnn_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

def train(model,train_loader,epochs):
    params=model.params()
    loss_fn=F.cross_entropy()
    optimizer=optim.Adam(params,lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    train_loss=[]
    for epoch in range(epochs):
        loss=0
        for i,data in enumerate(train_loader):
            inputs,labels=data
            outputs=model(inputs)
            curr_loss=loss_fn(outputs,labels)
            curr_loss.backward()
            optimizer.step()
            loss+=curr_loss.item()
        loss=np.mean(loss)
        train_loss.append(loss)
        print("Epoch : %d  Loss : %.5f",epoch,loss)

        if epoch%10==0:
            print("Saving the model")
            torch.save(model,"./model_epoch_{}_loss_{}".format(epoch,loss))


    return model


def test(model,test_loader):
    correct=0
    total=0
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            inputs,labels=data

            outputs=model(inputs)
            _,predicted=torch.max(outputs,1)
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network %d %%",(100*correct/total))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
	                help="path to h5 File containing data")
    ap.add_argument("-e", "--epochs",type=int,default="",
            help="Enter the number of epochs you want model to train")
    ap.add_argument("-b","--batch_size",type=int,default="",help="Enter the batch size for train set")

    args = vars(ap.parse_args())

    if not args.get("input"):
        raise argparse.ArgumentError("h5 file path isn't specified")
    elif not args.get("epochs"):
        raise argparse.ArgumentError("Epochs field should not be empty")
    elif not args.get("batch_size"):
        raise argparse.ArgumentError("Batch size should not be empty")
    else:
        X,Y= retrieve_data(args.get("input"))
        n=len(X)
        train_loader,test_loader=prepare_dataset(X,Y,args.get("batch_size"))
        model=patch_cnn_model()
        model=train(model,train_loader,args.get("epochs"))
        test(model,test_loader)






