import os
import sys
import einops
import time
import torch
import torch.nn as nn
from data.data import testDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model.models import model_vit1
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    model_dict = torch.load(r'./output/models/GAI_vit1.pth')
    model = model_vit1()
    model.load_state_dict(model_dict['model'])
    model.to(device)
    valImageDataset = testDataset(annotationsFile = "label/test_sample.csv",imgDir = "images/test",angle = 450,
        transform=ToTensor(),time = True)
    valDataLoader = DataLoader(valImageDataset,batch_size=4,shuffle = True)
    model.eval()
    predict = np.array([])
    real = np.array([])
    sites = []
    val_loss = 0
    criterion=nn.L1Loss()
    with torch.no_grad():
        for i, data in enumerate(valDataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = model(inputs)
            predict = np.append(predict, outputs.transpose(-1,0).cpu().detach().numpy())
            real = np.append(real,labels.cpu().detach().numpy())
        # sites.extend(site)
            print("predict:")
            print(predict)
            print("real:")
            print(real)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()

    val_loss = val_loss
    print('\tLoss: {:.6f}'.format(val_loss))
    csv = [predict, real]
    df = pd.DataFrame({'predict':predict,'real':real})
    df.to_csv(r'./output/results/res.csv')

if __name__ == "__main__":
    main()
