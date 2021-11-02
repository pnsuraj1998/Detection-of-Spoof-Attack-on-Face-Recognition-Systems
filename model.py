import torch.nn as nn
import torch.nn.functional as F
import torch

class patch_cnn_model(nn.Module):
    def __init__(self) -> None:
        super(patch_cnn_model,self).__init__()
        self.conv_1=nn.Conv2d(in_channels=3,out_channels=50,kernel_size=5,stride=1,padding='same')
        self.norm_1=nn.BatchNorm2d(num_features=50)
        self.pool_1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv_2=nn.Conv2d(in_channels=50,out_channels=100,kernel_size=3,stride=1,padding='same')
        self.norm_2=nn.BatchNorm2d(num_features=100)
        self.pool_2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv_3=nn.Conv2d(in_channels=100,out_channels=150,kernel_size=3,stride=1)
        self.norm_3=nn.BatchNorm2d(num_features=150)
        self.pool_3=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv_4=nn.Conv2d(in_channels=150,out_channels=200,kernel_size=3,stride=1)
        self.norm_4=nn.BatchNorm2d(num_features=200)
        self.pool_4=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv_5=nn.Conv2d(in_channels=200,out_channels=250,kernel_size=3,stride=1)
        self.norm_5=nn.BatchNorm2d(num_features=250)
        self.pool_5=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc_1=nn.Linear(in_features=2250,out_features=1000)
        self.dropout=nn.Dropout(p=0.5)
        self.fc_2=nn.Linear(in_features=1000,out_features=400)
        self.fc_3=nn.Linear(in_features=400,out_features=2)

    def forward(self,X):
        X=F.relu(self.conv_1(X))
        X=self.pool_1(self.norm_1(X))

        X=F.relu(self.conv_2(X))
        X=self.pool_2(self.norm_2(X))

        X=F.relu(self.conv_3(X))
        X=self.pool_3(self.norm_3(X))

        X=F.relu(self.conv_4(X))
        X=self.pool_4(self.norm_4(X))

        X=F.relu(self.conv_5(X))
        X=self.pool_5(self.norm_5(X))

        X=torch.flatten(X,1)
        
        X=F.relu(self.fc_1(X))
        X=self.dropout(X)
        X=F.relu(self.fc_2(X))
        X=F.softmax(self.fc_3(X))

        return X

if __name__=="__main__":
    model=patch_cnn_model()
    print(model)
    for parameter in model.parameters():
        print(parameter)

