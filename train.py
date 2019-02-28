# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:55:10 2018

@author: emily
"""

import model
import ut
import tensorflow as tf
import read_training_dataset
import os
import siamese as siam
path = "./tiger.jpeg"
pic_size1 = 127#123
pic_size2 = 255#251




#以下是一个月之前调试所注释掉的部分
#net = model.Vgg19()
#batch1 = ut.gen_batch(pic_size1,path)
#batch2 = ut.gen_batch(pic_size2,path)
#
#feature1 = net.build(batch1,pic_size1,reuse = False)
#feature2 = net.build(batch2,pic_size2,reuse = True)
#
#filt = tf.reshape(feature1,[123,123,4,1])
#output = tf.nn.conv2d(feature2,filt,strides=[1,1,1,1],padding='VALID')
#scaled_output = tf.image.resize_bilinear(output,[255,255])
#print(scaled_output)
#
# read tfrecodfile holding all the training data


#下面两行生成了batch为8的tfrecord
#data_reader = read_training_dataset.myReader(700, 700, 3)
#batched_data = data_reader.read_tfrecord(os.path.join("tfrecords", "training_dataset"), num_epochs = 50, batch_size = 8)    



#下面开始调试fcsiamese源代码
#hp, evaluation, run, env, design = parse_arguments()
## Set size for use with tf.image.resize_images with align_corners=True.
## For example,
##   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
## instead of
## [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
#final_score_sz = hp.response_up * (design.score_sz - 1) + 1
#    
# build the computational graph of Siamese fully-convolutional network
siamNet = siam.Siamese(batch_size = 8)
# get tensors that will be used during training
siamNet.build_tracking_graph_train()
#image, z_crops, x_crops, templates_z, scores, loss, train_step, distance_to_gt, summary= siamNet.build_tracking_graph_train(final_score_sz, design, env, hp)
# 
## read tfrecodfile holding all the training data
#data_reader = src.read_training_dataset.myReader(design.resize_width, design.resize_height, design.channel)
#batched_data = data_reader.read_tfrecord(os.path.join(env.tfrecord_path, env.tfrecord_filename), num_epochs = design.num_epochs, batch_size = design.batch_size)
#    
## run trainer
#trainer(hp, run, design, final_score_sz, batched_data, image, templates_z, scores, loss, train_step, distance_to_gt,  z_crops, x_crops, siamNet, summary)
#
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
