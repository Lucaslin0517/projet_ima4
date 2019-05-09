import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def load_data(input_file):
    ''' load data '''
    print("Load Data:\n")
    dataset = np.loadtxt(input_file, delimiter=",")
    print("Done")
    print(len(dataset))
    return dataset

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
    s1= ','.join(str(n) for n in x)
    file = open('DQN_result/x.txt','w')
    file.write(str(s1))
    file.close
    s2= ','.join(str(n) for n in StaY)
    file = open('DQN_result/y.txt','w')
    file.write(str(s2))
    file.close
    # plt.plot(x,StaA,color=(0,0.5,1),label="VelA")

    plt.xlabel("Time(/0.1s)")
    plt.ylabel("Velctory")
    # plt.xlabel("Time(s)")
    # plt.ylabel("Volt")
    plt.title("PyPlot Data")


    # plt.ylim(-1.5,1.5)
    plt.legend()
    # plt.xticks(range(-9,9))
    # plt.yticks(range(-6,6))
    plt.xticks(range(-8,9))
    plt.yticks(range(0,11))
   # plt.show()


class DQN_env:

    def find_maze_matrix(self, input_file):
        img1 = cv2.imread(input_file, 0)
        print(type(img1))
        print(img1.shape)
        r_map = []
        n = 0
        box = None
        for i in range(10, 200, 20):
            for j in range(10, 320, 20):
                print(img1[i, j])
                if img1[i, j] == 0:
                    box = 1
                else:
                    box = 0

                r_map.append(box)
                n += 1
        print(n)
        r_map_matrix = np.array(r_map)
        r_map_matrix = r_map_matrix.reshape((10,16))
        # r_map_list = list(r_map_matrix)
        print(r_map_matrix)
        return r_map_matrix

    def get_pos(self, input_file):
        dataset = load_data(input_file)
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
            x = math.floor(x)+8
            y = math.floor(y)
        
            if coord != [x, y]:
                coord_set.append([x, y])
            coord = [x, y]
        coord_set = np.array(coord_set)
        sta_P = coord_set[0]
        tar_P = coord_set[-1]
        sta_P = (sta_P[1],sta_P[0])
        tar_P = (tar_P[1],tar_P[0])

        print(coord_set[:,0])
        print(coord_set[:,1])
        plt.plot(coord_set[:,0],coord_set[:,1],color=(1,0,1),label="path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("box_place")
        # plt.xticks(range(-9,9))
        # plt.yticks(range(-6,6)) 
        plt.xticks(range(0,17))
        plt.yticks(range(0,11)) 
       # plt.show()
        plot(dataset)
        
        return sta_P, tar_P

    def get_pos_pred(self):
        sta_P = [-4.00000000000,6.00000000000]
        tar_P = [-2.61569978131,0.72902305453]
        sta_A = 2.9137539841
        tar_A = 0.241124508132
        sta_P[0] = math.floor(sta_P[0])+8
        sta_P[1] = math.floor(sta_P[1])
        tar_P[0] = math.floor(tar_P[0])+8
        tar_P[1] = math.floor(tar_P[1])
        # print(sta_P)
        # print(tar_P)
        start_P = (sta_P[1], sta_P[0])
        parger_P = (tar_P[1], tar_P[0])
        sta_P_0 = (-4.00000000000,6.00000000000)
        tar_P_0 = (-2.61569978131,0.72902305453)
        return sta_A, tar_A, sta_P_0, tar_P_0, start_P, parger_P
