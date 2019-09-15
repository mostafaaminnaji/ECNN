#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# C)Mostafa Amin-Naji, Babol Noshirvani University of Technology,
# My Official Website: www.Amin-Naji.com
# My Email: Mostafa.Amin.Naji@Gmail.com

# PLEASE CITE THE BELOW PAPER IF YOU USE THIS CODE

# M. Amin-Naji, A. Aghagolzadeh, and M. Ezoji, “Ensemble of CNN for Multi-Focus Image Fusion”, Information Fusion, vol. 51, pp. 21–214, 2019. 
# DOI: https://doi.org/10.1016/j.inffus.2019.02.003


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import numpy as numpy
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
import imageio


# In[ ]:


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.1, inplace=True),

        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.1, inplace=True),

        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.1, inplace=True),

        )
        
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )   
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )   
        
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )   
        
        
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )   
    
        
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )   

        self.fc1 = nn.Linear(256*8*4*2, 2)

        
        
        
    def forward(self, x, y, z):
        outx = self.conv1_1(x)
        outx = self.conv2_2(outx)
        outx = self.conv3_1(outx)
        outx = self.conv4(outx)
        outx = outx.view(outx.size(0), -1)
        
        outy = self.conv1_2(y)
        outy = self.conv2_2(outy)
        outy = self.conv3_2(outy)

        
        outz = self.conv1_3(z)
        outz = self.conv2_3(outz)
        outz = self.conv3_3(outz)
        
        oyz=torch.cat([outy,outz],1)
        
        oyz = self.conv5(oyz)
        oyz = oyz.view(oyz.size(0), -1)
        
        
        
        oo=torch.cat([outx,oyz],1)
        

        
        out = self.fc1(oo)
        
       

   
        return out


# In[ ]:


model=CNN()
model


# In[ ]:


model_path='ECNN_trained_network_wights.pth'

use_gpu=torch.cuda.is_available()

if use_gpu:

    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.load_state_dict(torch.load(model_path))
    
else:
    
    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path,map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


# In[ ]:



original_path1= './lytro-03-A.jpg'  
original_path2= './lytro-03-B.jpg'


# In[ ]:


# Define the window size

windows_size=32
# it can be set as 2 or 4 based on the size of input images, stride=2 is more accurate but stride=4 is more faster
stride = 2  


tfms1 = transforms.Compose([
    transforms.Resize((64, 32)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.45 ], [0.1])
])

tfms2 = transforms.Compose([
    transforms.Resize((64, 32)),
    transforms.ToTensor(),
    transforms.Normalize([ 0.050], [ 0.09])
])
tfms3 = transforms.Compose([
    transforms.Resize((64, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.06], [ 0.09])
])


img1 = Image.open(original_path1)
img2 = Image.open(original_path2)

img1 = np.asarray(img1)
img2 = np.asarray(img2)

kernel=np.array([[-1 ,   -2 ,   -1],  [0 ,    0    , 0],  [1 ,    2,     1]])
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_GY = cv2.filter2D(img1_gray,-1,kernel)
img1_GX = cv2.filter2D(img1_gray,-1,np.transpose(kernel))

img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_GY = cv2.filter2D(img2_gray,-1,kernel)
img2_GX = cv2.filter2D(img2_gray,-1,np.transpose(kernel))


# test_image1 = cv2.resize(test_image1, (test_image1.shape[1]*1, test_image1.shape[0]*1)) 
# test_image2 = cv2.resize(test_image2, (test_image2.shape[1]*1, test_image2.shape[0]*1))

test_image1_1=img1_gray
test_image1_2=img1_GX
test_image1_3=img1_GY

test_image2_1=img2_gray
test_image2_2=img2_GX
test_image2_3=img2_GY


source1=img1
source2=img2


j=0

map1=np.zeros([img1.shape[0], img1.shape[1]])
map2=np.zeros([img1.shape[0], img1.shape[1]])
map3=np.zeros([img1.shape[0], img1.shape[1]])
map4=map3
MAP=np.zeros([img1.shape[0], img1.shape[1]])

score1=0
score2=0
FUSED=np.zeros(test_image1_1.shape)

windowsize_r = windows_size-1
windowsize_c = windows_size-1

for r in tqdm(range(0,img1.shape[0] - windowsize_r, stride)):
    for c in range(0,img1.shape[1] - windowsize_c, stride):
        
        block_test1_1 = test_image1_1[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test1_2 = test_image1_2[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test1_3 = test_image1_3[r:r+windowsize_r+1,c:c+windowsize_c+1]
        
        block_test2_1 = test_image2_1[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test2_2 = test_image2_2[r:r+windowsize_r+1,c:c+windowsize_c+1]
        block_test2_3 = test_image2_3[r:r+windowsize_r+1,c:c+windowsize_c+1]
 
        block1_1= np.concatenate((block_test1_1, block_test2_1), axis=0)
        block2_1= np.concatenate((block_test2_1, block_test1_1), axis=0)  
        block1_1 = Image.fromarray(block1_1, 'L')
        block2_1 = Image.fromarray(block2_1, 'L')
        block1_2= np.concatenate((block_test1_2, block_test2_2), axis=0)
        block2_2= np.concatenate((block_test2_2, block_test1_2), axis=0)  
        block1_2 = Image.fromarray(block1_2, 'L')
        block2_2 = Image.fromarray(block2_2, 'L')
        block1_3= np.concatenate((block_test1_3, block_test2_3), axis=0)
        block2_3= np.concatenate((block_test2_3, block_test1_3), axis=0)  
        block1_3 = Image.fromarray(block1_3, 'L')
        block2_3 = Image.fromarray(block2_3, 'L')
                 
        imout1_1=tfms1(block1_1)
        imout2_1=tfms1(block2_1)
        imout1_2=tfms2(block1_2)
        imout2_2=tfms2(block2_2)
        imout1_3=tfms3(block1_3)
        imout2_3=tfms3(block2_3)
        
        if use_gpu:
            imout1_1=to_var(imout1_1)
            imout2_1=to_var(imout2_1)
            imout1_2=to_var(imout1_2)
            imout2_2=to_var(imout2_2)
            imout1_3=to_var(imout1_3)
            imout2_3=to_var(imout2_3)
        
        imout1_1=(imout1_1)
        imout2_1=(imout2_1)
        imout1_2=(imout1_2)
        imout2_2=(imout2_2)
        imout1_3=(imout1_3)
        imout2_3=(imout2_3)
        
        
        inputs1_1 = imout1_1.unsqueeze(0)
        inputs2_1 = imout2_1.unsqueeze(0)
        inputs1_2 = imout1_2.unsqueeze(0)
        inputs2_2 = imout2_2.unsqueeze(0)
        inputs1_3 = imout1_3.unsqueeze(0)
        inputs2_3 = imout2_3.unsqueeze(0)

        model.eval()

        outputs1 = model(inputs1_1,inputs1_2,inputs1_3)
        _, predicted1 = torch.max(outputs1.data, 1)
        
        score1=predicted1.detach().cpu().numpy()

        model.eval()
        
        outputs2 = model(inputs2_1,inputs2_2,inputs2_3)
        _, predicted2 = torch.max(outputs2.data, 1)
        
        score2=predicted2.detach().cpu().numpy()
        
        map2[r:r+windowsize_r+1,c:c+windowsize_c+1] += 1
        
        if score1 <= score2:
            map1[r:r+windowsize_r+1,c:c+windowsize_c+1] += +1 
      
        else:
            map1[r:r+windowsize_r+1,c:c+windowsize_c+1] += -1 
            


# In[ ]:


plt.imshow(map1, cm.gray)


# In[ ]:


map11=map1
img1 = imageio.imread(original_path1)
img2 = imageio.imread(original_path2)

test_image1 = np.asarray(img1)
test_image2 = np.asarray(img2)

test_image1 = rgb2gray(test_image1)
test_image2 = rgb2gray(test_image2)

FUSED=np.zeros(img1.shape)
for r in range(0,img1.shape[0], 1):
    for c in range(0,img1.shape[1], 1):   
        
        if map11[r,c] < 0:
            map3[r,c] =0
            FUSED[r,c]=img2[r,c]
            
        else:
            map3[r,c] =1
            FUSED[r,c]=img1[r,c]
            
FUSED_8=FUSED.astype(np.uint8)
# imageio.imwrite('./output.tif', FUSED_8)
# imageio.imwrite('./output.jpg', FUSED_8)
plt.imshow(map3, cm.gray)


# In[ ]:


plt.imshow(FUSED_8)


# In[ ]:


# x=test_image1.shape[0]//20
# z=test_image1.shape[1]//20
# kernel = np.ones((x,z),np.float32)/(x*z)
# MAP2 = cv2.filter2D(map1,-1,kernel)
# MAP2[MAP2<0.5] = 0
# MAP2[MAP2>=0.5] = 1

# FUSED_CV=np.zeros(img1.shape)

# FUSED=np.zeros(img1.shape)
# for r in range(0,img1.shape[0], 1):
#     for c in range(0,img1.shape[1], 1):   
        
#         if MAP2[r,c] < 0.5:
            
#             FUSED_CV[r,c]=img1[r,c]
            
#         else:
            
#             FUSED_CV[r,c]=img2[r,c]


# FUSED_CV=FUSED_CV.astype(np.uint8)
# imageio.imwrite('./output_CV.tif', FUSED_CV)
# imageio.imwrite('./output_CV.jpg', FUSED_CV)

# plt.imshow(MAP2, cm.gray)


# In[ ]:




