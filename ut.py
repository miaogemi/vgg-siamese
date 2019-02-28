# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 01:34:29 2018

@author: emily
"""
import skimage
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import ut
import tensorflow as tf
#输入图片路径和要求图像大小，输出要求图像大小（pic_size,pic_size,3）的数组，但是范围在【0,1】
def load_image(path,pic_size):
    # load image
    img = skimage.io.imread(path)
    img = img/255.0
#    print ("Original Image Shape: ", img.shape)
    #print(np.shape(img))
    # we crop image from center
    short_edge = min(img.shape[:2])#前两维最小值
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]#从中间切出最小维度大小的方块
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (pic_size,pic_size))
    #print(np.shape(resized_img))
#    plt.imshow(resized_img)
#    plt.show()
    return resized_img
#print(load_image('./tiger.jpeg',227))
    
def gen_batch(pic_size,path):
    img = ut.load_image(path,pic_size)
    batch = img.reshape((1, pic_size, pic_size, 3))
    batch = tf.cast(batch,dtype=tf.float32)
    return batch