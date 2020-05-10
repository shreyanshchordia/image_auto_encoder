#===========================================================
# importing libraries

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
from torchsummary import summary

#===========================================================
# Model Architecture 

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 16, 5, 5

            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  #16, 5, 5
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # 8, 15, 15
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encodings(self, x):
        e = self.encoder(x)
        return e

#===========================================================
# To turn tensors into images

def to_img(x):
    x = (0.5 * x) + 0.5
    return (x.view(x.shape[0],1,28,28))

#===========================================================
# Function to train model

def train_model(model,dataloader,num_epochs,cost,optimizer):

    loss_values = []

    for epoch in range(1,num_epochs+1):

        for data in dataloader:
            img, _ = data
            img = Variable(img).cuda()

            # =================== forward =====================
            output = model(img)
            loss = cost(output, img)
            # =================== backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # =================== log ========================


        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, 
                                                 num_epochs, 
                                                  loss.data))
        loss_values.append(loss.data)

        if(epoch==1 or epoch%10==0):
            pic = to_img(output.cpu().data)
            save_image(pic, '/content/drive/My Drive/Colab Notebooks/stage/epoch_{}.png'.format(epoch))
        
    return loss_values
   
#===========================================================
# Required tranforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std= (0.5))
])

#===========================================================
# Defined parameters

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001

#===========================================================
# Dataset and DataLoader

train_dataset = MNIST('./data',
                      train = True,
                      transform = img_transform,
                      download = True)

train_dataloader = DataLoader(train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              num_workers = 4)

test_dataset = MNIST('./data',
                      train = False,
                      transform = img_transform,
                      download = True)

test_dataloader = DataLoader(test_dataset,
                              batch_size = BATCH_SIZE//2,
                              shuffle = True)

#===========================================================
# Compiling the model

model = autoencoder().cuda()
cost = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = LEARNING_RATE,
                             weight_decay=1e-5)

#===========================================================
# Training the model

train_model(model,train_dataloader,EPOCHS,cost,optimizer)

#===========================================================
# Saving the model

torch.save(model.state_dict(),"/auto_encoder_conv.pth")
