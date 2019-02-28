import tensorflow as tf
import numpy as np

_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)

#input
#._create_siamese_train(x_crops= size(24, 255, 255, 3), z_crops=size(8, 127, 127, 3),
# h=[11, 5, 3, 3, 3],w=[11, 5, 3, 3, 3],num=[96, 128, 96, 96, 32])

#output template_z, templates_x = #(8,17,17,32) [24, 49, 49, 32]
def create_siamese_train(self, net_x, net_z,h,w,num):
    filter_h = h
    filter_w = w
    filter_num = num

		# loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
    for i in range(_num_layers):
        print('> Layer '+str(i+1))
		
			####close group
        _filtergroup_yn = np.array([0,0,0,0,0], dtype=bool)
			# set up conv "block" with bnorm and activation 
        net_x = set_convolutional_train(net_x, filter_h[i], filter_w[i], filter_num[i], _conv_stride[i],
				                filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
				                scope='conv'+str(i+1), reuse=False)
		
			# notice reuse=True for Siamese parameters sharing
        net_z = set_convolutional_train(net_z, filter_h[i], filter_w[i], filter_num[i],_conv_stride[i],
				                filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
				                scope='conv'+str(i+1), reuse=True)    
		
			# add max pool if required
        if _pool_stride[i]>0:
            print('\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i]))
            net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
            net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))

    return net_z, net_x#输出一个batch所有的图片经过alexnet的tensor结果


def set_convolutional_train(X, filter_h, filter_w, filter_num, stride, filtergroup=False, batchnorm=True,
                      activation=True, scope=None, reuse=True):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv', reuse=reuse):
        input_channel = X.get_shape().as_list()[-1]
        # sanity check    
        W = tf.get_variable("W", shape = [filter_h, filter_w,
                                input_channel / (2 if filtergroup else 1), filter_num])
        b = tf.get_variable("b", [1, W.get_shape().as_list()[-1]])

        if filtergroup:
            X0, X1 = tf.split(X, 2, 3)
            W0, W1 = tf.split(W, 2, 3)
            h0 = tf.nn.conv2d(X0, W0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(X1, W1, strides=[1, stride, stride, 1], padding='VALID')
            
            h = tf.concat([h0, h1], 3) + b
        else:
            h = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID') + b

        if batchnorm:
            h = tf.layers.batch_normalization(h)

        if activation:
            h = tf.nn.relu(h)

        return h