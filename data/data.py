import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import random
import torchvision.transforms as transforms

import numpy as np

import os
import sys

class realDataset(Dataset):
    def __init__(self,annotationsFile,imgDir,transform=None,
                 targetTransform = None,angle=0,time=False):
        self.img_labels = pd.read_csv(annotationsFile)
        self.img_dir = imgDir
        self.transform = transform
        self.target_transform = targetTransform
        self.angle = angle
        self.time=time
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        site = self.img_labels.iloc[idx,0]
        site = site.rstrip()
        date = self.img_labels.iloc[idx,1]
        Nitrogen = self.img_labels.iloc[idx,6]
        Cultivar = str(self.img_labels.iloc[idx,4])
        Cultivar = Cultivar.rstrip()
        Replicate = str(self.img_labels.iloc[idx,7])
        if(Replicate[0]>='0' and Replicate[0]<='9'):
            Replicate = Replicate[0]
        tt = str(int(self.img_labels.iloc[idx,2]))
        year=''
        day=''
        month=''
        monthpos = 0
        pos = 0
        for ch in date:
            if ch == '/':
                if  year == '':
                    year = date[0:pos]
                    monthpos = pos+1
                elif month == '':
                    month = date[monthpos:pos]
                    day=date[pos+1:].rstrip()
                    break
            pos+=1
        if(len(month)<2):
            month='0'+month
        if(len(day)<2):
            day='0'+day
        date = date.rstrip()
        if date=="2021/3/18":
            day = '17'
        if date=="2021/1/8":
            day = '11'
        plot = str(self.img_labels.iloc[idx,5]).rstrip()
        img_path_0 = ''
        img_path_45 = ''
        if site == 'Yangling':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-1.jpg')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-3.jpg')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-1.png')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-3.png')
        elif site == 'Baima':
            if Nitrogen == 'N3':
                Nitrogen = 'N2'
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+Nitrogen+'-'+Replicate+'/'+plot+'_0.png')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+Nitrogen+'-'+Replicate+'/'+plot+'_45.png')
        elif site == 'Jurong':
            i=0
            j=0
            while (not (os.path.exists(img_path_0)) and i<10):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_0/'+'Jurong-'+year+month+day+'-'+Cultivar+Nitrogen+'-rep'+str(i)+'-DSC-RX0r.png')
                i += 1
            while (not (os.path.exists(img_path_45)) and j<10):
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_45/'+'Jurong-'+year+month+day+'-'+Cultivar+Nitrogen+'-rep'+str(j)+'-DSC-RX045.png')
                j += 1
        elif site == 'Xuzhou':
            i=0
            j=0
            while (not (os.path.exists(img_path_0)) and i<10):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'Xuzhou-'+year+month+day+'-'+plot+Nitrogen+'-rep'+str(i)+'-DSC-RX0r.png')
                i += 1
            while (not (os.path.exists(img_path_45)) and j<10):
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_45/'+'Xuzhou-'+year+month+day+'-'+plot+Nitrogen+'-rep'+str(j)+'-DSC-RX045.png')
                j += 1
        elif site == 'France':
            i=0
            j=0
            while (not (os.path.exists(img_path_0)) and i<10):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'.'+month+'.'+day+'/'+'Plot_'+plot+'/Camera2_'+str(i)+'_GRefWB.jpg')
                i += 1
            while (not (os.path.exists(img_path_45)) and j<10):
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'.'+month+'.'+day+'/'+'Plot_'+plot+'/Camera1_'+str(j)+'_GRefWB.jpg')
                j += 1
        elif site == 'Soil':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+plot+'_0.JPG')
            img_path_45 = os.path.join(self.img_dir, site + '/' + year + '_' + month + '_' + day + '/' + plot + '_45.JPG')
        elif site == 'Sim2real':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+plot+'_'+tt+'_220_45_scale1.jpg')
            img_path_45 = os.path.join(self.img_dir, site + '/' + year + '_' + month + '_' + day + '/' + plot + '_' + tt + '_220_45_scale1.jpg')
        else:
            print(site,"Wrong!")
        time = 0
        if self.time:
            time = self.img_labels.iloc[idx,9]
        train_augs = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)])
        img_0 = read_image(img_path_0)
        img_45 = read_image(img_path_45)
        _, img_0_h, img_0_w = img_0.shape
        _, img_45_h, img_45_w = img_45.shape
        img_size = min(min(img_0_w, img_0_h), img_45_w, img_45_h)
        image_0 = transforms.Resize((384, 384))(img_0)
        image_45 = transforms.Resize((384, 384))(img_45)
        if img_size > 384:
            s0 = min(min(img_0_h, img_0_w) // 2, img_size)
            s45 = min(img_45_w // 2, img_size)
            width_crop_0 = (img_0_w - s0) // 2
            height_crop_0 = (img_0_h - s0) // 2
            side0 = (img_0_h - s0 * 2) // 2
            wide0 = (img_0_w - s0 * 2) // 2
            width_crop_45 = (img_45_w - s45) // 2
            height_crop_45 = (img_45_h - s45) // 2
            side45 = (img_45_h - s45 * 2) // 2
            wide45 = (img_45_w - s45 * 2) // 2
            new_img0_1 = img_0[:, height_crop_0:img_0_h - height_crop_0, width_crop_0:img_0_w - width_crop_0]
            new_img0_2 = img_0[:, side0:img_0_h - s0 - side0, wide0:img_0_w - wide0 - s0]
            new_img0_3 = img_0[:, img_0_h - side0 - s0:img_0_h - side0, img_0_w - wide0 - s0:img_0_w - wide0]
            new_img45_1 = img_45[:, height_crop_45:img_45_h - height_crop_45, width_crop_45:img_45_w - width_crop_45]
            new_img45_2 = img_45[:, height_crop_45:img_45_h - height_crop_45, wide45:img_45_w - wide45 - s45]
            new_img45_3 = img_45[:, height_crop_45:img_45_h - height_crop_45, img_45_w - wide45 - s45:img_45_w - wide45]
            rd = (random.randint(1, 10000) % 3) + 1
            new_0 = new_img0_1
            if rd == 2:
                new_0 = new_img0_2
            elif rd == 3:
                new_0 = new_img0_3
            new_45 = new_img45_1
            if rd == 2:
                new_45 = new_img45_2
            elif rd == 3:
                new_45 = new_img45_3
            image_0 = transforms.Resize((384, 384))(new_0)
            image_45 = transforms.Resize((384, 384))(new_45)
        image_0_45 = torch.cat((image_0,image_45),0)
        image_0 = train_augs(image_0)
        image_45 = train_augs(image_45)
        image_0_45 = torch.cat((image_0, image_45), 0)
        image=np.array([])
        label = torch.tensor(self.img_labels.iloc[idx,8])

        if self.angle == 0:
            image =  image_0
        elif self.angle == 45:
            image =  image_45
        elif self.angle ==450:
            image =  image_0_45
        else:
            print('wrong angle')

        image = image.numpy()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.time:
            return (np.swapaxes(image,0,1),torch.full([1],int(time))),(label.float())
        else:
            return np.swapaxes(image,0,1).to(device='cuda').to(device='cuda'),(label.float().to(device='cuda'))

class testDataset(Dataset):
    def __init__(self,annotationsFile,imgDir,transform=None,
            targetTransform = None,angle=0,time=False):
        self.img_labels = pd.read_csv(annotationsFile)
        self.img_dir = imgDir
        self.transform = transform
        self.target_transform = targetTransform
        self.angle = angle
        self.time=time
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        site = self.img_labels.iloc[idx,0]
        site = site.rstrip()
        date = self.img_labels.iloc[idx,1]
        Nitrogen = self.img_labels.iloc[idx,6]
        Replicate = str(self.img_labels.iloc[idx,7])
        Cultivar = str(self.img_labels.iloc[idx,4])
        Cultivar = Cultivar.rstrip()
        year=''
        day=''
        month=''
        monthpos = 0
        pos = 0
        for ch in date:
            if ch == '/':
                if  year == '':
                    year = date[0:pos]
                    monthpos = pos+1
                elif month == '':
                    month = date[monthpos:pos]
                    day=date[pos+1:].rstrip()
                    break
            pos+=1
        if(len(month)<2):
            month='0'+month
        if(len(day)<2):
            day='0'+day
        date = date.rstrip()
        if date=="2021/3/18":
            day = '17'
        if date=="2021/1/8":
            day = '11'
        plot = str(self.img_labels.iloc[idx,5]).rstrip()
        img_path_0 = ''
        img_path_45 = ''
        if site == 'Yangling':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-1.jpg')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-3.jpg')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-1.png')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-3.png')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'_0/'+plot+'.JPG')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'_45/'+plot+'.JPG')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_0/'+plot+'.JPG')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_45/'+plot+'.JPG')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-1.jpg')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'/'+'uplot_'+plot+'_1'+'/Image-1-cam-3.jpg')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,year+month+day+'/45/'+plot+'#'+str(Nitrogen)+'.JPG')
                img_path_45 = os.path.join(self.img_dir,year+month+day+'/45/'+plot+'#'+str(Nitrogen)+'.JPG')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'/0/'+plot+'.jpg')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'/45/'+plot+'_45.jpg')

        elif site == 'Baima':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(Cultivar)+str(Nitrogen)+'-'+plot+'_0.JPG')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(Cultivar)+str(Nitrogen)+'-'+plot+'_45.JPG')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(Cultivar)+str(Nitrogen)+'-'+plot+'-0.JPG')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(Cultivar)+str(Nitrogen)+'-'+plot+'-45.JPG')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(Cultivar)+str(Nitrogen)+'_'+plot+'_0.JPG')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(Cultivar)+str(Nitrogen)+'_'+plot+'_45.JPG')
        
        elif site == 'Jurong':
            i=0
            j=0
            while (not (os.path.exists(img_path_0)) and i<10):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_0/'+'Jurong-'+year+month+day+'-'+Cultivar+Nitrogen+'-rep'+str(i)+'-DSC-RX0r.png')
                i += 1
            while (not (os.path.exists(img_path_45)) and j<10):
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_45/'+'Jurong-'+year+month+day+'-'+Cultivar+Nitrogen+'-rep'+str(j)+'-DSC-RX045.png')
                j += 1

        elif site == 'Xuzhou':
            i=0
            j=0
            while (not (os.path.exists(img_path_0)) and i<10):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_0/'+'Xuzhou-'+year+month+day+'-'+Cultivar+Nitrogen+'-rep'+str(i)+'-DSC-RX0r.png')
                i += 1
            while (not (os.path.exists(img_path_45)) and j<10):
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_45/'+'Xuzhou-'+year+month+day+'-'+Cultivar+Nitrogen+'-rep'+str(j)+'-DSC-RX045.png')
                j += 1
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_0/'+Cultivar+Nitrogen+'-rep0-DSC-RX0r.JPG')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'_45/'+Cultivar+Nitrogen+'-rep0-DSC-RX045.JPG')       
        
        elif site == 'France':
            i=0
            j=0
            while (not (os.path.exists(img_path_0)) and i<10):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'.'+month+'.'+day+'/'+'Plot_'+plot+'/Camera2_'+str(i)+'_GRefWB.jpg')
                i += 1
            while (not (os.path.exists(img_path_45)) and j<10):
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'.'+month+'.'+day+'/'+'Plot_'+plot+'/Camera1_'+str(j)+'_GRefWB.jpg')
                j += 1
        
        elif site == 'Yuanyang':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/0/'+'1-'+year+month+day+'-'+plot+'#'+str(Nitrogen)+'-rep0-DSC-RX0.JPG')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/45/'+'1-'+year+month+day+'-'+plot+'#'+str(Nitrogen)+'-rep0-DSC-RX0.JPG')
        
        elif site == 'JiNan':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/0/'+plot+'.JPG')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/45/'+plot+'.JPG')
        
        elif site == 'Yangzhou':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+str(plot)+'-'+str(Nitrogen)+'.jpg')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'_'+month+'_'+day+'/'+str(plot)+'-'+str(Nitrogen)+'.jpg')
        
        elif site == 'Zhengzhou':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(plot)+'-'+str(Nitrogen)+'.jpg')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+month+day+'/'+str(plot)+'-'+str(Nitrogen)+'.jpg')
        
        elif site == 'Xinxiang':
            img_path_0 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'/'+plot+'-RX0.JPG')
            img_path_45 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'/'+plot+'-RX045.JPG')
            if (not os.path.exists(img_path_0)):
                img_path_0 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'-0/'+str(Cultivar)+'#'+str(Nitrogen)+'.JPG')
                img_path_45 = os.path.join(self.img_dir,site+'/'+year+'-'+month+'-'+day+'-45/'+str(Cultivar)+'#'+str(Nitrogen)+'.JPG')
        
        else:
            print(site,"Wrong!")

        time = 0
        if self.time:
            time = self.img_labels.iloc[idx,2]
        img_0 = read_image(img_path_0)
        img_45 = read_image(img_path_45)
        _, img_0_h, img_0_w = img_0.shape
        _, img_45_h, img_45_w = img_45.shape
        img_size = min(min(img_0_w, img_0_h), img_45_w, img_45_h)
        width_crop_0 = (img_0_w - img_size) // 2
        height_crop_0 = (img_0_h - img_size) // 2
        width_crop_45 = (img_45_w - img_size) // 2
        height_crop_45 = (img_45_h - img_size) // 2
        new_img_0 = img_0[:, height_crop_0:img_0_h - height_crop_0, width_crop_0:img_0_w - width_crop_0]
        new_img_45 = img_45[:, height_crop_45:img_45_h - height_crop_45, width_crop_45:img_45_w - width_crop_45]
        image_0 = transforms.Resize((384, 384))(new_img_0)
        image_45 = transforms.Resize((384, 384))(new_img_45)
        image_0_45 = torch.cat((image_0,image_45),0)
        image=np.array([])
        label = torch.tensor(self.img_labels.iloc[idx,8])

        if self.angle == 0:
            image =  image_0
        elif self.angle == 45:
            image =  image_45
        elif self.angle ==450:
            image =  image_0_45
        else:
            print('wrong angle')

        image= image.numpy()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.time:
            return (np.swapaxes(image,0,1).to(device='cuda'),torch.full([1],int(time)).to(device='cuda')),(label.float().to(device='cuda'))
        else:
            return np.swapaxes(image,0,1).to(device='cuda').to(device='cuda'),(label.float().to(device='cuda'))
