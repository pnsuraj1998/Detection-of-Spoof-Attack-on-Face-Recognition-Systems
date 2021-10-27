import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

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









def retrieve_data(csv_file):
    hf = h5py.File('data.h5', 'r')
    data = hf.get('Dataset_Data')
    labels=hf.get('Dataset_Labels')
    return np.array(data),np.array(labels)
