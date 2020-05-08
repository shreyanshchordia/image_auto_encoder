import numpy as np
import tensorflow as tf
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):

    def __init__(self):
        
        super(AutoEncoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features= 784,out_features= 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),

            nn.Linear(in_features=128, out_features = 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),

            nn.Linear(in_features=64, out_features = 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=16, out_features = 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1),

            nn.Linear(in_features= 64,out_features= 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),

            nn.Linear(in_features=128, out_features = 784),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2)
        )
    
    def forward(self, x):

        encoded = self.encoder(x)

        x = self.decoder(encoded)

        return x
    
    def get_encodings(self, x):

        encoded = self.encoder(x)

        return encoded

def to_img(x):
    x = (0.5 * x) + 0.5
    return (x.view(x.shape[0],1,28,28))

def train_model(model,dataloader,num_epochs,cost,optimizer):

    loss_values = []

    for epoch in range(1,num_epochs+1):

        for data in dataloader:
            img, _ = data
            img = Variable(img).cuda()

            # =================== Forward =====================
            output = model(img)
            loss = cost(output, img)
            # =================== Backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # =================== Log =========================


        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, 
                                                  num_epochs, 
                                                  loss.data))
        loss_values.append(loss.data)

        if epoch==1 or epoch % 5==0 :
            pic = to_img(output.cpu().data)
            save_image(pic, './ae_img/epoch_{}.png'.format(epoch))
        
    return loss_values

# ================= Transformers ======================

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std= (0.5)),
    transforms.Lambda(lambda x: torch.flatten(x))
])

# ================= Parameters ========================

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# ================= Dataset ===========================

train_dataset = MNIST('./data',
                      train = True,
                      transform = img_transform,
                      download = True)

train_dataloader = DataLoader(train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              num_workers = 2)

test_dataset = MNIST('./data',
                      train = False,
                      transform = img_transform,
                      download = True)

test_dataloader = DataLoader(test_dataset,
                              batch_size = BATCH_SIZE//2,
                              shuffle = True)

# =============== Initializing =========================

model = AutoEncoder().cuda()
cost = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 
                            lr = LEARNING_RATE, 
                            weight_decay=1e-5)

# ================= Training ===========================

train_model(model,train_dataloader,EPOCHS,cost,optimizer)

# ================ Saving Model ======================

torch.save(model.state_dict(), './autoencoder.pth')
