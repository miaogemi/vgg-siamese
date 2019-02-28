# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:57:48 2018

@author: emily
"""

#import tensorflow as tf
import numpy as np
#import model
#import ut
import tensorflow as tf
#import read_training_dataset
#import os
#import siamese as siam
#import skimage
#import skimage.io
#import cv2
#path = "./tiger.jpg"
###
##initial = [[[[1.,1.],[2.,2.]],[[1.,1.],[2.,2.]]],[[[1.,1.],[2.,2.]],[[1.,1.],[2.,2.]]],[[[1.,1.],[2.,2.]],[[1.,1.],[2.,2.]]]]
x = np.arange(24*3*3)

score = np.reshape(x,(24,3,3,1))
print(score)#[24, 255, 255, 1]
s = tf.squeeze(tf.stack([score[i]  for i in [0 + 3 * i for i in range(8)]]))
print("shape of score map:", s.get_shape().as_list())
##print(np.shape(initial))
#x = tf.Variable(initial,dtype=tf.float32)
#print(np.shape(x))
#b = tf.shape(x)[1:]
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
#    print(sess.run(tf.reduce_mean(x)))
#    print(sess.run(x))
#    print(tf.shape(x))
    print(sess.run(s))
#    print(x[1].get_shape())
###import tensorflow as tf
###import numpy as np
###
###initial = [[1.,1.],[2.,2.]]
###x = tf.Variable(initial,dtype=tf.float32)
###init_op = tf.global_variables_initializer()
###with tf.Session() as sess:
###    sess.run(init_op)
###    print(sess.run(tf.reduce_mean(x)))
###    print(sess.run(tf.reduce_mean(x,0))) #Column
###    print(sess.run(tf.reduce_mean(x,(0,1)))) #row

##测试crop.pad函数
#pic = ut.load_image(path,700)
#im = tf.Variable(pic)
#avg_chan = tf.reduce_mean(im,axis=(0,1))
##print(avg_chan.get_shape())
#npad = 20
#paddings = [[npad, npad], [npad, npad], [0, 0]]
#cha = im - avg_chan#图像像素减去平均像素
#im_padded = tf.pad(cha, paddings, mode='CONSTANT')
#im_new = im_padded + avg_chan
#
#
##测试extract x函数
#a = extract_crops_x(im_new,npad,6,9,400,400,420,255)
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
##    print(np.shape(mean),np.shape(im_new))
##    print(sess.run(avg_chan[0]))
##    print(sess.run(cha[0,0,0]))
##    print(sess.run(im[0,0,0]))
##    print(sess.run(im_padded[0,0,0]))
##    print(sess.run(im_new[0,0,0]))
#    print(a.get_shape())