import os
import sys
import cv2

img_path = '/mnt/d/backup/resnet50_sfp/imagenet'
gt_path = '/mnt/d/backup/Vit_B_16_ImageNet1k/matlab_npz/image'


def read_path(img_path):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(img_path):
        #print(filename)
        img = cv2.imread(img_path+'/'+filename)
        if img.shape[2] > 3:
            img = img[:,:,:3]
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB,)
        new_image = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
        #####保存图片的路径
        cv2.imwrite(gt_path+"/"+filename, new_image)


read_path(img_path)
#print(os.getcwd())
