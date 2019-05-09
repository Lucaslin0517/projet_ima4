import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re
import os.path
import math
import cv2

# from keras.utils import to_categorical


def load_data(input_file):
    ''' load data '''
    print("Load Data:\n")
    dataset = np.loadtxt(input_file, delimiter=",")
    print("Done")
    print(len(dataset))
    return dataset


def readwrite(input_file,output_file):
    data_f=pd.read_csv(input_file,names=['StaX', 'StaY', 'StaA', 
                                         'VelX', 'VelY', 'VelA',
                                         'EndX', 'EndY', 'EndA', 'image'],sep=',')
    print(type(data_f), '\t', data_f.shape)
    # print(data_f[[0]].shape)
    data_f[['StaX', 'StaY', 'StaA', 
            'VelX', 'VelY', 'VelA',
            'EndX', 'EndY', 'EndA']].to_csv(output_file, sep=',', header=False,index=False)
    

def getRunTimes( fun ,input_file,output_file,fun_name):
    begin_time=int(round(time.time() * 1000))
    fun(input_file,output_file)
    end_time=int(round(time.time() * 1000))
    print(fun_name,(end_time-begin_time),"ms")


def plot(data):
    # x = np.linspace(0,len(data)-1,len(data))
    x = data[:,0]
    StaX = data[:,0]
    StaY = data[:,1]
    StaA = data[:,3]
    VelX = data[:,3]
    VelY = data[:,4]
    VelA = data[:,5]
    EndX = data[:,6]
    EndY = data[:,7]
    EndA = data[:,8]
    # VelA = data[:,5]

    # plt.figure(figsize=(8,4))

    # plt.plot(x,StaX,label="VelX",color="red",linewidth=2)
    plt.plot(x,StaY,color=(0,1,1),label="VelY")
    # plt.plot(x,StaA,color=(0,0.5,1),label="VelA")

    plt.xlabel("Time(/0.1s)")
    plt.ylabel("Velctory")
    # plt.xlabel("Time(s)")
    # plt.ylabel("Volt")
    plt.title("PyPlot Data")


    # plt.ylim(-1.5,1.5)
    plt.legend()
    plt.xticks(range(-9,9))
    plt.yticks(range(-6,6))
    plt.show()


def load_imdata(input_file):
    image_datas = []
    X_im = []
    pathDir_1 =  os.listdir(input_file)
    for allDir_1 in pathDir_1:
        child = os.path.join('%s/%s' % (input_file, allDir_1))
        df=pd.read_csv(child)
        image_datas.append(df)
    for data in image_datas:
        # print(data.shape)
        x = data.values.reshape(-1, data.shape[0], data.shape[1], 1).astype('float32') / 255.
        X_im.append(x)
    print(len(X_im))
    return X_im






if __name__ == "__main__":
    # getRunTimes(readwrite,'telemetry.csv','tm_data.csv', "transfer csv：") 
    # # readwrite2('telemetry.csv', 'tm_data.csv')
    dataset = load_data('full_data/1/data/190314_14_28_52.csv')
    StaX = dataset[:,0]
    StaY = dataset[:,1]
    StaA = dataset[:,3]
    box_num_set = []
    box_num = None
    print(StaY)
    coord = None
    coord_set = []
    for i in range(len(StaX)):
        x = StaX[i]
        y = StaY[i]
        x = math.floor(x)+0.5
        y = math.floor(y)+0.5
        if coord != [x, y]:
            coord_set.append([x, y])
        coord = [x, y]
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,5]:
        #         box_num = 9+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,4]:
        #         box_num = 25+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,3]:
        #         box_num = 41+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,2]:
        #         box_num = 57+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,1]:
        #         box_num = 73+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,0]:
        #         box_num = 89+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,-1]:
        #         box_num = 105+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,-2]:
        #         box_num = 121+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,-3]:
        #         box_num = 137+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,-4]:
        #         box_num = 153+i
        # for i in range(-8,7):
        #     print(i)
        #     if coord == [i,-4]:
        #         box_num = 169+i
        
    START_P = (dataset[0,0],dataset[0,1])
    TARGET_P = (dataset[-1,0],dataset[-1,1])
    print(START_P)
    print(TARGET_P)
    coord_set = np.array(coord_set)
    print(coord_set[:,0])
    print(coord_set[:,1])
    plt.plot(coord_set[:,0],coord_set[:,1],color=(0,1,1),label="path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("box_place")
    plt.xticks(range(-9,9))
    plt.yticks(range(-6,6))
    plt.show()
    plot(dataset)


    # img1 = cv2.imread("1-10/1/data_20190314_14_28_52/map/map_000000001.png", 0)
    # print(type(img1))
    # print(img1.shape)
    # r_map = []
    # n = 0
    # box = None
    # for i in range(10, 200, 20):
    #     for j in range(10, 320, 20):
    #         print(img1[i, j])
    #         if img1[i, j] == 0:
    #             box = 1
    #         else:
    #             box = 0

    #         r_map.append(box)
    #         n += 1
    # print(n)
    # r_map_matrix = np.array(r_map)
    # r_map_matrix = r_map_matrix.reshape((10,16))
    # r_map_list = list(r_map_matrix)
    # print(r_map_list)

    # x_imdata = load_imdata('images_data') 

    




