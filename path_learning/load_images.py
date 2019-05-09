import os.path
import re
from PIL import Image
import numpy
import cv2



def load_image(input_file):
    # pathDir_1 =  os.listdir(input_file)
    # for allDir_1 in pathDir_1:
    #     child = os.path.join('%s\%s' % (input_file, allDir_1))
    #     # print(allDir_1)
    #     # print(child)
    #     pathDir_2 = os.listdir(child)
    #     for allDir_2 in pathDir_2:
    #         child_txt = os.path.join('%s\%s' % (child, allDir_2))
    #         # print(child_txt)
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
                if os.path.isfile(child_txt_2) == False:
                    # print(os.path.isfile(child_txt))
                    # print(child_txt[14:31])
                    # print(child_txt)
                    pathDir_3 = os.listdir(child_txt_2)
                    # print(len(pathDir_3))
                    allDir_3 = pathDir_3[0]
                    child_images = os.path.join('%s/%s' % (child_txt_2, allDir_3))
                    img1 = cv2.imread(child_images, 0)
                    img1_shape = img1.shape
                    print(img1_shape)
                    img_ndarray = img1
                    # name = child_images[14:29]
                    name = child_images[24:36]
                    print(name)
                    # numpy.savetxt('full_data/' + allDir_1 + '/images_data/'  + str(name) + '.csv', img_ndarray, delimiter = ',')
                    numpy.savetxt('predict_data/' + allDir_1 + '/images_data/'  + str(name) + '.csv', img_ndarray, delimiter = ',')



if __name__ == '__main__':
    # load_image('LIN_CHEN')
    load_image('predict_source_data')
