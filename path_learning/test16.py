import numpy as np
import pandas as pd
import os
import copy
import random
import cv2
import matplotlib;

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt;
import math
from dqn_maze import DQN_env
from PIL import Image

filenamex = 'DQN_result/posx_all.txt'
filenamey = 'DQN_result/posy_all.txt'

origin_filex = 'DQN_result/x.txt'
origin_filey = 'DQN_result/y.txt'


posx = []
posy = []
origin_px = []
origin_py = []

def load_data(input_file):
  dataset = []
  with open(input_file, 'r') as file_to_read:# read the posx 
    while True:
      lines = file_to_read.readline() # read data
      if not lines:
        break
        pass
      p_tmp = [float(i) for i in lines.split(',')]
      dataset.append(p_tmp) 
  return dataset[0]

posx = load_data(filenamex)
posy = load_data(filenamey)
origin_px = load_data(origin_filex)
origin_py = load_data(origin_filey)

im = Image.open("1-10/8/data_20190314_16_09_44/map/map_000000002.png")


img = np.array(im)
data = pd.DataFrame(img) #use pandas handle the data

'''
--->>>change

for i in range(0,200):
  for j in range(0,320):
    if data.at[i,j] > 2 and data.at[i,j]<160:  
      data.at[i,j] = 64
    elif data.at[i,j] >= 160:
      data.at[i,j] = 255
    else:
      data.at[i,j] = 0


--->>change
'''
for i in range(0,200):
  for j in range(0,320):
    if data.at[i,j] != 0:
      data.at[i,j] = 255  
     

origin_color = 128
predi_color = 192
'''
for i in range(0,len(origin_px)):
  raw = (int(200-(origin_py[i]-0.5)*20))#  0~200
  col = (int((origin_px[i]+7.5)*20))#make - to +  0~320
  data.at[raw,col] = origin_color
'''
  


for i in range(0,len(posx)):
  raw1 = (int(200-(posy[i]-0.5)*20))
  col1 = (int((posx[i]+7.5)*20))
  data.at[raw1,col1] = predi_color
  if(i+1 != len(posx)):
    raw2 = (int(200-(posy[i+1]-0.5)*20))
    col2 = (int((posx[i+1]+7.5)*20))
    if col2 > col1 and raw2 == raw1:
      k = col1
      while k <= col2:
        data.at[raw1,k] = predi_color
        k = k+1
    elif col2 < col1 and raw2 == raw1:
      k = col2
      while k <= col1:
        data.at[raw1,k] = predi_color
        k = k+1
    elif col1 == col2 and raw2 > raw1:
      k = raw1
      while k <= raw2:
        data.at[k,col1] = predi_color
        k = k+1
    elif col1 == col2 and raw1 > raw2:
      k = raw2
      while k <= raw1:
        data.at[k,col1] = predi_color
        k = k+1 
    elif col1 < col2 and raw1 <raw2:
      k1 = raw1
      k2 = col1
      while k1 <= raw2 and k2 <= col2:
        data.at[k1,k2] = predi_color
        k1 = k1+1
        k2 = k2+1
    elif col2 < col1 and raw2 <raw1:
      k1 = raw2
      k2 = col2
      while k1 <= raw1 and k2 <= col1:
        data.at[k1,k2] = predi_color
        k1 = k1+1
        k2 = k2+1
    elif col1 < col2 and raw1 > raw2:
       k1 = col1
       k2 = raw2
       while k1 <= raw1 and k2 <= col1:
         data.at[k2,k1] = predi_color
         k1 = k1+1
         k2 = k2+1
    elif col2 < col1 and raw2 > raw1:
       k1 = col2
       k2 = raw1
       while k1 <= raw1 and k2 <= col1:
         data.at[k2,k1] = predi_color
         k1 = k1+1
         k2 = k2+1  

'''
for i in range(0,len(posx)):
  raw = (int(200-(posy[i]-0.5)*20))#  0~200
  col = (int((posx[i]+7.5)*20))#make - to +  0~320
  data.at[raw,col] = predi_color
'''

print("wait a moment")

for i in range(0,100):
  for j in range(0,320):
    data.ix[i,j],data.ix[200-1-i,j] = data.ix[200-1-i,j],data.ix[i,j]

#->>colors u can insert the color 

colors = ['black','gray','white','blue','white'] #grad protect  red origin  blue predi
cmap = matplotlib.colors.ListedColormap(colors,'indexed')#0~255 if u divided colors into 3  0->black 255/2->red 255->white   if u divieded colors into 4  the same ... 

#cmap='gray'
img=np.array(data[0:])
im.close()

plt.imshow(img,cmap=cmap)
plt.show()


