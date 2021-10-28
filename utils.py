import h5py,torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

class create_Dataset(Dataset):
    def __init__(self,data,labels,transform=None):
        self.data=data
        self.labels=labels
        self.transform=transforms.Compose([transforms.ToTensor()
                                               ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        x,y=self.data[idx],torch.tensor(self.labels[idx])
        if self.transform:
            x=self.transform(x)
        y=F.one_hot(y,num_classes=2)
        return x,y
def prepare_dataset(X,Y,batch):
    n=len(X)
    dataset=create_Dataset(X,Y,transform=None)
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,(int(n*0.8),int((n*0.2))))
    train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=1)
    return train_loader,test_loader

def retrieve_data(csv_file):
    hf = h5py.File('data.h5', 'r')
    data = hf.get('Dataset_Data')
    labels=hf.get('Dataset_Labels')
    return np.array(data),np.array(labels)
