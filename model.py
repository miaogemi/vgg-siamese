# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:33:25 2018

@author: emily
"""

import tensorflow as tf

import os
import cv2
import numpy as np
import inspect
import ut
from functools import reduce
VGG_MEAN = [103.939, 116.779, 123.68]#训练集的平均值，后期要修改

class Vgg19:
    def __init__(self, vgg19_npy_path=None,trainable = True):
        self.trainable = trainable
        self.var_dict = {}

        #生成一个vgg19参数文件的路径
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)# 输出D:\test\vgg-siamese\vgg19.npy
            
            #load读取读取npy参数文件，文件格式为数组，数组内容是字典，类似np.array({'cov1':'【123】'},dtype=object)
            #item（）在内容为字典的情况下，可以逐条读取字典内容，但在np.array([1,2,3]）情况下不行
            #items()可以直接遍历读取字典内容
            #功能：读取参数文件，产生{conv1_1：[参数值]}字典
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
            print("npy file loaded")
            #a = self.data_dict['conv1_1'][0](3,3,3,64)weight值
            #b = self.data_dict['conv1_1'][1]（64）bias值
            #print(self.data_dict['conv1_1'][0])
            #print(self.data_dict['conv1_1'])#可以查看每层参数值           
            
    def build(self, rgb,pic_size,reuse):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """


        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        #tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，
        #如果是0就表示对第0维度进行切割。num_split就是切割的数量
        #red(1,254,254,1)把rgb层分开表示
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        #这部分断言可以不用，原vgg19输入图像大小规定为224
        #我的网络输入图像大小有变化，所以直接设定pic_size传进来
        assert red.get_shape().as_list()[1:] == [pic_size, pic_size, 1]
        assert green.get_shape().as_list()[1:] == [pic_size, pic_size, 1]
        assert blue.get_shape().as_list()[1:] == [pic_size, pic_size, 1]
        #训练集的平均值，后期要根据自己的训练集进行修改
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [pic_size, pic_size, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1",reuse =reuse)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2",reuse =reuse)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1",reuse =reuse)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2",reuse =reuse)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1",reuse =reuse)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2",reuse =reuse)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3",reuse =reuse)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4",reuse =reuse)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1",reuse =reuse)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2",reuse =reuse)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3",reuse =reuse)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4",reuse =reuse)
#        self.pool4 = self.max_pool(self.conv4_4, 'pool4')
#不再使用第五卷积层，因为网络太深了，产生出的图片特征向量尺寸已经小于1
#        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
#        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
#        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
##        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
#        print(np.shape(self.conv1_1))
#        print(np.shape(self.conv2_2))
#        print(np.shape(self.conv3_4))
#        print(np.shape(self.conv2_2))
#        print(np.shape(self.conv4_4))
        
        ##将conv1-2,2-2,3-4,4-4取出来 各自经过1*1卷积层
        with tf.variable_scope('conv1',reuse=reuse):
            conv1_weight = tf.get_variable('weight',[1,1,64,1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv1_bias = tf.get_variable('bias',[1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv = tf.nn.conv2d(self.conv1_2,conv1_weight,strides=[1,1,1,1],padding='VALID')
            self.conv1 = tf.nn.bias_add(conv,conv1_bias)#276
#        print(np.shape(self.conv1))
        with tf.variable_scope('conv2',reuse=reuse):
            conv2_weight = tf.get_variable('weight',[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv2_bias = tf.get_variable('bias',[1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv = tf.nn.conv2d(self.conv2_2,conv2_weight,strides=[1,1,1,1],padding='VALID')
            self.conv2 = tf.nn.bias_add(conv,conv2_bias)
        with tf.variable_scope('conv3',reuse=reuse):
            conv3_weight = tf.get_variable('weight',[1,1,256,1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv3_bias = tf.get_variable('bias',[1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv = tf.nn.conv2d(self.conv3_4,conv3_weight,strides=[1,1,1,1],padding='VALID')
            self.conv3 = tf.nn.bias_add(conv,conv3_bias)
        with tf.variable_scope('conv4',reuse=reuse):
            conv4_weight = tf.get_variable('weight',[1,1,512,1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv4_bias = tf.get_variable('bias',[1],initializer=tf.truncated_normal_initializer(stddev=0.001))
            conv = tf.nn.conv2d(self.conv4_4,conv4_weight,strides=[1,1,1,1],padding='VALID')
            self.conv4 = tf.nn.bias_add(conv,conv4_bias)
#        print(np.shape(conv4))
        
        
        ##upsampling操作 全都变成conv1大小276
        feature_size = tf.cast(self.conv1.shape[1],tf.int32)
        binimg1 = tf.image.resize_bilinear(self.conv1,(feature_size,feature_size))
        binimg2 = tf.image.resize_bilinear(self.conv2,(feature_size,feature_size))
        binimg3 = tf.image.resize_bilinear(self.conv3,(feature_size,feature_size))
        binimg4 = tf.image.resize_bilinear(self.conv4,(feature_size,feature_size))
#        print(feature_size)
        ##concatenate
        concat = tf.concat([binimg1,binimg2,binimg3,binimg4],axis=3)
        print(concat)
        
        
        self.data_dict = None
        return concat
        

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)
    
    
#self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
#self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
    def conv_layer(self, bottom, in_channels, out_channels, name,reuse):
        with tf.variable_scope(name,reuse=reuse):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

# filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

#filters = self.get_var(initial_value, nameconv1_1, 0, name + "_filters")
#biases = self.get_var(initial_value, name, 1, name + "_biases")
    def get_var(self, initial_value, name, idx, var_name):
        #数据字典加载完成后并且name在字典里存在，就赋值
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        #如果网络是可训练的，那把value赋值给ternsor变量，如果不可训练，赋值为常量
        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
    
    def loss(scoremap,label):#scoremap:互相关层上采样至原图大小的scoremap值，label就是标签值  
        loss = tf.reduce_mean(tf.log(1 + tf.exp(-scoremap * label)))
        return loss


          

