import numpy as np
import pandas as pd
import os
from collections import deque
from sklearn.utils import shuffle
from keras.losses import mean_squared_error
import copy
import random
from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten
import cv2
import matplotlib.pyplot as plt
import math
from dqn_maze import DQN_env

env = DQN_env()
maze = env.find_maze_matrix("1-10/8/data_20190314_16_09_44/map/map_000000002.png") #"1-10/2/data_20190314_15_12_25/map/map_000000001.png"
sta_P, tar_P = env.get_pos('full_data/1/data/190314_14_36_22.csv')
_, _, _, _, sta_P_pred, tar_P_pred = env.get_pos_pred()
# sta_P = np.array(sta_P)
# tar_P = np.array(tar_P)
print(sta_P_pred)
print(tar_P_pred)

# model_name = 'dqn_model.h5'
model_name = 'dqn_model_demo.h5'


TMP_VALUE = 2

# start_state_pos = (5,5)
start_state_pos = sta_P
# start_state_pos = sta_P_pred

# target_state_pos = (6,6)
target_state_pos = tar_P
# target_state_pos = tar_P_pred

actions = dict(
    up = 0,
    down = 1,
    left = 2,
    right = 3
)

action_dimention = len(actions)

reward_dict = {'reward_0': -1, 'reward_1': -0.01, 'reward_2': 1}



def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):
            if type(i)== list:
                input_list = i + input_list[index+1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break
    return output_list





def matrix_to_img(row,col):
    state = copy.deepcopy(maze)
    state[row, col] = TMP_VALUE
   
    state = np.reshape(state,newshape=(1, state.shape[0],state.shape[1],1))
    return state

class DQNAgent:
    def __init__(self,agent_model=None):
        self.memory = deque(maxlen=100)
        self.alpha = 0.01
        self.gamma = 0.9  # decay rate
        self.epsilon = 1
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        if agent_model is None:
            self.model = self.dqn_model()
        else:
            self.model = agent_model

    def dqn_model(self):
        inputs = Input(shape=(maze.shape[0], maze.shape[1],1))
        layer1 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same')(inputs)
        layer2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer1)
        layer3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer2)
        layer4 = Flatten()(layer3)
        predictions = Dense(action_dimention, activation='softmax')(layer4)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='sgd',
                      loss=mean_squared_error,
                      )
        return model


    def remember(self,current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

   
    def choose_action(self, state):
       
        if np.random.rand() < self.epsilon:
            action = random.choice(list(actions.keys()))
            action = actions.get(action)
            return action
       
        else:
            act_values = self.model.predict(state)
           
            #action = np.idmax(shuffle(pd.Series(act_values[0])))
            action = np.argmax(shuffle(pd.Series(act_values[0])))
            return action

    def repay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        batch_random_choice = np.random.choice(len(self.memory),batch_size)
        for i in batch_random_choice:
            current_state, action, reward, next_state, done = self.memory[i]


            target_f = self.model.predict(current_state)
            if done:
                target = reward
            else:
                target = reward + self.alpha * (self.gamma * np.max(self.model.predict(next_state)[0]) - target_f[0][action])
            target_f[0][action] = target


            self.model.fit(current_state, target_f, nb_epoch=2, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min


class Environ:
    def __init__(self):
        pass

    def step(self,current_state, action):
       
        row, col = np.argwhere(current_state == TMP_VALUE)[0,1:3]
        done = False
        if action == actions.get('up'):
            next_state_pos = (row - 1, col)
        elif action == actions.get('down'):
            next_state_pos = (row + 1, col)
        elif action == actions.get('left'):
            next_state_pos = (row, col - 1)
        else:
            next_state_pos = (row, col + 1)
        if next_state_pos[0] < 0 or next_state_pos[0] >= maze.shape[0] or next_state_pos[1] < 0 or next_state_pos[1] >= maze.shape[1] \
                or maze[next_state_pos[0], next_state_pos[1]] == 1:
           
            next_state = copy.deepcopy(current_state)
            reward = reward_dict.get('reward_0')

        elif next_state_pos == target_state_pos: 
            next_state = matrix_to_img(target_state_pos[0],target_state_pos[1])
            reward = reward_dict.get('reward_2')
            done = True
        else:  # maze[next_state[0],next_state[1]] == 0
            next_state = matrix_to_img(next_state_pos[0], next_state_pos[1])
            reward = reward_dict.get('reward_1')
        return next_state, reward, done


def train():
    if os.path.exists(model_name):
        agent_model = load_model(model_name)
        agent = DQNAgent(agent_model=agent_model)
    else:
        agent = DQNAgent()
 
    environ = Environ()

    episodes = 2000
    for e in range(episodes):
       
        current_state = matrix_to_img(start_state_pos[0],start_state_pos[1])

        i = 0
        while(True):
            i = i + 1
           
            action = agent.choose_action(current_state)
           
            next_state, reward, done= environ.step(current_state,action)
           
            agent.remember(current_state, action, reward, next_state, done)
            if done:
               
                print("episode: {}, step used:{}" .format(e,  i))
                break

            current_state = copy.deepcopy(next_state)
           
            if i % 100 == 0:
                agent.repay(100)
          
        if (e+1) % 200 == 0:
            agent.model.save(model_name)


def predict():
    pos_set = [start_state_pos]
    
    actions_new = dict(zip(actions.values(),actions.keys()))
   
    agent_model = load_model(model_name)
    environ = Environ()
    current_state = matrix_to_img(start_state_pos[0], start_state_pos[1])

    for i in range(100):

        action = agent_model.predict(current_state)
        
        action = np.argmax(action[0])

        next_state, reward, done = environ.step(current_state, action)
        print('current_state: {}, action: {}, next_state: {}'.format(np.argwhere(current_state==TMP_VALUE)[0,1:3], actions_new[action], np.argwhere(next_state==TMP_VALUE)[0,1:3]))
        next_pos = (np.argwhere(next_state==TMP_VALUE)[0,1:3][0],np.argwhere(next_state==TMP_VALUE)[0,1:3][1])
        pos_set.append(next_pos)
       
        if done:
            break
        
        current_state = next_state
    print(pos_set)
    return pos_set
        


if __name__ == "__main__":

    #train()

    pos_set = predict()
    
    sta_A, tar_A, sta_P_0, tar_P_0, sta_P_pred, tar_P_pred = env.get_pos_pred()
    tem_x = [sta_P_0[0]]
    tem_y = [sta_P_0[1]]
    for i in pos_set:
        
        tem_x.append(float(i[1])-7.5)
        tem_y.append(float(i[0])+0.5)
    tem_x.append(tar_P_0[0])
    tem_y.append(tar_P_0[1])
    # if tem_x[0] > tem_x[1] and tem_x[0] < tem_x[2]:
    #     tem_x.pop(1)
    #     tem_y.pop(1)
    # if tem_x[0] < tem_x[1] and tem_x[0] > tem_x[2]:
    #     tem_x.pop(1)
    #     tem_y.pop(1)
    tem_x.pop(1)
    tem_x.pop(-2)
    tem_y.pop(1)
    tem_y.pop(-2)
    print(tem_x)
    print(tem_y)
    
    posx_all = []
    posy_all = []
    for i in range(len(tem_x)):
        if i == len(tem_x)-1:
            break
        temx_new = np.linspace(tem_x[i],tem_x[i+1],36) 
        if tem_x[i]==tem_x[i+1]:
            temy_new = np.linspace(tem_y[i], tem_y[i+1],36)
        else:
            z1 = np.polyfit([tem_x[i], tem_x[i+1]], [tem_y[i], tem_y[i+1]], 1)
            temy_new = temx_new*z1[0]+z1[1] 
        
        temx_new = temx_new.tolist()
        temy_new = temy_new.tolist()
        posx_all.append(temx_new)
        posy_all.append(temy_new)
    posx_all = flatten(posx_all)
    posy_all = flatten(posy_all)
    # posx_all = np.array(posx_all)
    # posy_all = np.array(posy_all)
    # posx_all = posx_all.reshape((-1,1))
    print(posx_all)
    print(posy_all)
    file = open('DQN_result/posx_all.txt','w')
    file.write(str(posx_all))
    file.close
    file = open('DQN_result/posy_all.txt','w')
    file.write(str(posy_all))
    file.close
    plt.plot(posx_all,posy_all,color=(1,0,0),label="path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("pred_pos")
    # plt.xticks(range(-9,9))
    # plt.yticks(range(-6,6)) 
    plt.xticks(range(-8,9))
    plt.yticks(range(0,11)) 
    #plt.show()

    vx = []
    vy = []
    va = []
    dva = (tar_A - sta_A) / len(posx_all)
    for i in range(len(posx_all)):
        if i == len(posx_all)-1:
            break
        v = (posx_all[i+1]-posx_all[i])/0.01
        vx.append(v)
        v = (posy_all[i+1]-posy_all[i])/0.01
        vy.append(v)
        v = dva/0.01
        va.append(v)
        print(v)
    vx.append(0)
    vy.append(0)
    va.append(0)
    t = rxs = np.arange(0, len(vx)) / 100
    print(len(t))
    print(len(va))
    plt.plot(t,vx,'r')
    plt.plot(t,vy,'b')
    plt.plot(t,va,'y')
    #plt.show()
    file = open('DQN_result/V_X.txt','w')
    file.write(str(vx))
    file.close()
    file = open('DQN_result/V_Y.txt','w')
    file.write(str(vy))
    file.close()
    file = open('DQN_result/V_A.txt','w')
    file.write(str(va))
    file.close()
