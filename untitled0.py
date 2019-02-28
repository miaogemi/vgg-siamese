# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:37:14 2019

@author: emily
"""
import tensorflow as tf
var_dict = {}  
def get_var(layername, idx, var_name,data_dict):
#            #w = self.data_dict['conv1_1'][0](3,3,3,64)weight值
#            #b = self.data_dict['conv1_1'][1]（64）bias值
        #数据字典加载完成后并且name在字典里存在，就赋值
      

    value = data_dict[layername][idx]


        #如果网络是可训练的，那把value赋值给ternsor变量，如果不可训练，赋值为常量

    var = tf.Variable(value, name=var_name)

    var_dict[(layername, idx)] = var


    return var_dict,var

ddict = {'conv1': [[123],[456]], 'conv2': [[3214213],[146]]}
a,b = get_var('conv1',0,'conv1hh',ddict)
c,d = get_var('conv2',0,'conv2hh',ddict)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(c))
    print(c)

    


 