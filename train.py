import os
import sys
import einops
import time
import torch
from data.data import realDataset
from model.models import model_vit1

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    model = model_vit1()

    model = model.to(device)

    trainImageDataset = realDataset(annotationsFile="label/img_train.csv",imgDir="images/train",angle=450,
                      transform=ToTensor(),time=True)
    trainDataLoader = DataLoader(trainImageDataset,batch_size=4,shuffle = True,drop_last = False,num_workers=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=5*1e-6,weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer= optimizer,T_max=50,eta_min=1e-7)
    model.train()

    for epoch in range(200):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainDataLoader, 0):
            inputs, labels = data
            inputs = [inputs[0].to("cuda"), inputs[1].to("cuda")]
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 9 == 1 and i != 1 :
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                fi = open(r'./loss/train_loss.txt', 'a')
                fi.write(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}\n')
                fi.close()
                running_loss = 0.0
            if (epoch + 1) % 25 == 0 and (epoch+1) > 100 and (epoch + 1) != 150:
                cheakpoint = {
                    'model':model.module.state_dict()
                }
                torch.save(cheakpoint, r'./output/models/%sepoch.pth' % (str(epoch+1)))
            if (epoch + 1) % 10 == 0:
                cheakpoint = {
                    'model': model.module.state_dict()
                }
                torch.save(cheakpoint, r'./output/models/%sepoch.pth' % (str(epoch + 1)))
            scheduler.step()
        print('Finished Training')

if __name__ == "__main__":
    main()
