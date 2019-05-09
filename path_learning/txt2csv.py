import os.path
import re
import numpy
import pandas as pd
import csv


def readwrite(input_file,output_file):
    data_f=pd.read_csv(input_file,names=['StaX', 'StaY', 'StaA', 
                                         'VelX', 'VelY', 'VelA',
                                         'EndX', 'EndY', 'EndA', 'image'],sep=',')
    print(type(data_f), '\t', data_f.shape)
    # print(data_f[[0]].shape)
    data_f[['StaX', 'StaY', 'StaA', 
            'VelX', 'VelY', 'VelA',
            'EndX', 'EndY', 'EndA']].to_csv(output_file, sep=',', header=False,index=False)

def rw_vel(input_file,output_file):
    data_f=pd.read_csv(input_file,names=['StaX', 'StaY', 'StaA', 
                                         'VelX', 'VelY', 'VelA',
                                         'EndX', 'EndY', 'EndA', 'image'],sep=',')
    data_f[['VelX', 'VelY', 'VelA']].to_csv(output_file, sep=',', header=False,index=False)


def rw_sta(input_file,output_file):
    data_f=pd.read_csv(input_file,names=['StaX', 'StaY', 'StaA', 
                                         'VelX', 'VelY', 'VelA',
                                         'EndX', 'EndY', 'EndA', 'image'],sep=',')
    data_f[['StaX', 'StaY', 'StaA']].to_csv(output_file, sep=',', header=False,index=False)

def read_try(input_file):
    data_f=pd.read_csv(input_file)
    print(data_f)


def tran_csv(input_file):
    pathDir_1 =  os.listdir(input_file)
    for allDir_1 in pathDir_1:
        child = os.path.join('%s/%s' % (input_file, allDir_1))
        print(allDir_1)
        print(child)
        pathDir_2 = os.listdir(child)
        for allDir_2 in pathDir_2:
            child_txt = os.path.join('%s/%s' % (child, allDir_2))
            pathDir_3 = os.listdir(child_txt)
            for allDir_3 in pathDir_3:
                child_txt_2 = os.path.join('%s/%s' % (child_txt, allDir_3))
                if os.path.isfile(child_txt_2) and child_txt_2[-3:] == 'txt':
                    # print(child_txt_2)
                    # print(child_txt_2[14:29])
                    print(child_txt_2[24:36])
                    # read_try()
                    # readwrite(child_txt_2,'full_data/'+allDir_1+'/data/'+child_txt_2[14:29]+'.csv')
                    rw_vel(child_txt_2,'full_data/'+allDir_1+'/vel_data/'+child_txt_2[14:29]+'.csv')
                    # rw_sta(child_txt_2,'full_data/'+allDir_1+'/vel_data/'+child_txt_2[14:29]+'.csv')
                    # readwrite(child_txt_2,'predict_data/'+allDir_1+'/data/'+child_txt_2[24:36]+'.csv')
                    # rw_vel(child_txt_2,'predict_data/'+allDir_1+'/vel_data/'+child_txt_2[24:36]+'.csv')
                    
            


def fusion_csv(input_file):
    pathDir =  os.listdir(input_file)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (input_file, allDir))
        # print(child)
        csv_reader = csv.reader(open(child))
        for row in csv_reader:
            print(row)


if __name__ == '__main__':
    # tran_csv('LIN_CHEN')
    # tran_csv('predict_source_data')
    tran_csv('1-10')
    tran_csv('11-20')
    # fusion_csv('data')
