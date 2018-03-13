# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:30:40 2018

@author: lynden
squeezeNet for imageNet
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
import h5py

import utils

from datasets import dataset_factory
from preprocessing import preprocessing_factory

from Quantize import Quantizers
from Quantize import Factories 
from Quantize import QBatchNorm

slim = tf.contrib.slim

hdf5_path = './netData/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5'

# the dict for saving hdf5 weights name/values
hd5ModelDict = {}

# ==== hdf5 preprocessing ====
def hefVisitor_func(name, node):
    global hd5ModelDict
    if isinstance(node, h5py.Dataset):
        print("name %s is dataset" % name)
        print("name split %s" % str(name).split('/'))
        #print("name split %s" % str(name).split('_'))
        #print 'node shape = ', node.value.shape
        if str(name).split('/')[0] in ['fire2', 'fire3', 'fire4', 'fire5', \
                                       'fire6', 'fire7', 'fire8', 'fire9']:
            split = str(name).split('/')
            rename = split[0]+'/'+split[1]
            print '** rename = ', rename
            hd5ModelDict['squeezenet/'+rename] = node.value
        
        if str(name).split('/')[0] in ['conv1']:
            split = str(name).split('/')
            rename = split[0]
            print '** rename = ', rename
            hd5ModelDict['squeezenet/'+rename] = node.value
# ============================


# ==== Network model =========

def fire_module(x, fire_id, squeeze=16, expand=64,
                scope='fire',
                conv2d=slim.conv2d):
    scope = scope+str(fire_id)
    with tf.variable_scope(scope, [x]):   
        net = conv2d(x, squeeze, [1, 1],
                     padding='VALID',
                     activation_fn=tf.nn.relu,
                     scope='squeeze1x1')    
        left = conv2d(net, expand, [1, 1],
                      padding='VALID',
                      activation_fn=tf.nn.relu,
                      scope='expand1x1')
        right = conv2d(net, expand, [3, 3],
                       padding='SAME',
                       activation_fn=tf.nn.relu,
                       scope='expand3x3')
    
        net = tf.concat(axis=3, values=[left, right])
    return net

def squeezeNet(inputs, num_classes=1000,
               scope='squeezenet',
               conv2d=slim.conv2d, 
               max_pool2d=slim.max_pool2d):
    
    with tf.variable_scope(scope, 'squeezenet', [inputs, num_classes]):
        net = conv2d(inputs, 64, [3, 3], scope='conv1')
        net = max_pool2d(net, [2, 2], scope='pool1')
        
        net = fire_module(net, 2)
    
    return net

# the place to locate images
x = tf.placeholder(tf.float32, [5, 32, 32, 3])

# === read hdf5 ====
f5 = h5py.File(hdf5_path, 'r')

print("Keys: %s" % f5.keys())
f5.visititems(hefVisitor_func)
print hd5ModelDict.keys()    


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    net = squeezeNet(x)
    #softmax = tf.nn.softmax(net)
    
    vs = tf.trainable_variables()
    # load hdf5 weights into TF tensor
    for v in vs:
        print v
        name = str(v.name).split(':')[0]
        #print name
        #v.load(hd5ModelDict[name], sess)
        #assert np.all(sess.run(v) == hd5ModelDict[name])