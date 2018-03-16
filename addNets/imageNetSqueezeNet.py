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

from PIL import Image

import utils

from datasets import dataset_factory
from preprocessing import preprocessing_factory

from tensorflow.python.ops import math_ops

from Quantize import Quantizers
from Quantize import Factories 
from Quantize import QBatchNorm

slim = tf.contrib.slim

hdf5_path = './squeezeNet/netData/squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
#data_path   = '/runtmp3/lexhsu/imageDataset/imagenet_s/imagenet/val_src/s/val_0002'
#batch_size = 16 #1

tf.app.flags.DEFINE_string(
    'hdf5_path', '', 'Location of hdf5 weight file.')
tf.app.flags.DEFINE_string(
    'img_path', './squeezeNet/imgTest.npy', 'Location of image dataset.')
tf.app.flags.DEFINE_string(
    'lab_path', './squeezeNet/labTest.npy', 'Location of label dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1000, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'log_path', './squeezeNet/logTest', 'Location of log.')

#'qMap_cifar10Vgg16_fixed'
tf.app.flags.DEFINE_string(
    'intr_qmap', '', 'Location of intrinsic quantizer map.'
    'If empty, no quantizer is applied.')

tf.app.flags.DEFINE_string(
    'extr_qmap', '', 'Location of extrinsic quantizer map.'
    'If empty, no quantizer is applied.')

tf.app.flags.DEFINE_string(
    'weight_qmap', '', 'Location of weight quantizer map.'
    'If empty, no quantizer is applied.')

FLAGS = tf.app.flags.FLAGS

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
            rename = split[0]+'/'+split[1]+'/'+split[3]
            print '** rename = ', rename
            hd5ModelDict['squeezenet/'+rename] = node.value
        
        if str(name).split('/')[0] in ['conv1', 'conv10']:
            split = str(name).split('/')
            rename = split[0]+'/'+split[1]
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
        net = conv2d(inputs, 64, [3, 3], 2, padding='VALID', scope='conv1')
        net = max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool1')
        
        net = fire_module(net, fire_id=2, squeeze=16, expand=64, conv2d=conv2d)
        net = fire_module(net, fire_id=3, squeeze=16, expand=64, conv2d=conv2d)
        net = max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool3')
        
        net = fire_module(net, fire_id=4, squeeze=32, expand=128, conv2d=conv2d)
        net = fire_module(net, fire_id=5, squeeze=32, expand=128, conv2d=conv2d)
        net = max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool5')
        
        net = fire_module(net, fire_id=6, squeeze=48, expand=192, conv2d=conv2d)
        net = fire_module(net, fire_id=7, squeeze=48, expand=192, conv2d=conv2d)
        net = fire_module(net, fire_id=8, squeeze=64, expand=256, conv2d=conv2d)
        net = fire_module(net, fire_id=9, squeeze=64, expand=256, conv2d=conv2d)
        
        #net = slim.dropout(net, 0.5, scope='dropout9')
        net = conv2d(net, num_classes, [1, 1], padding='VALID', scope='conv10')
        # Global Average Pooling
        #net = tf.nn.avg_pool(net, ksize=(1, 13, 13, 1), strides=(1, 2, 2, 1), padding='SAME')
        #net = avg_pool2d(net, [13, 13], padding='VALID', stride=1, scope='avgpool10')
        #net = slim.flatten(net)
        #net = math_ops.reduce_mean(net, [1, 2], name='globalAvgPool1')
        net = tf.reduce_mean(net, [1, 2], name='globalAvgPool1')
    
    return net

# the place to locate images
x = tf.placeholder(tf.float32, [5, 227, 227, 3])

# === read hdf5 ====
f5 = h5py.File(hdf5_path, 'r')

print("Keys: %s" % f5.keys())
f5.visititems(hefVisitor_func)
print hd5ModelDict.keys()    


# === imageNet dataset processing ====
'''
# ==== prepare dataset ====
dataset = dataset_factory.get_dataset(
        'imagenet', 'validation', data_path)

print('dataset= %s' % dataset)

provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=8 * batch_size,
        common_queue_min=batch_size*4)

[image, label] = provider.get(['image', 'label'])

#####################################
# Select the preprocessing function #
#####################################
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    'vgg',
    is_training=False)

eval_image_size = 227
image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=5*batch_size)
'''

# ==== initial variables ====
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    start_time_simu = time.time()
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #imgs = np.load('imgTest.npy')
    #imgs, labs = sess.run([images, labels])
    
    imgs = np.load(FLAGS.img_path)
    labs = np.load(FLAGS.lab_path)
    # ==== image array value normalization ====
    #print imgs
    # 'RGB' -> 'BGR'   
    imgs = imgs[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    imgs[..., 0] -= mean[0]
    imgs[..., 1] -= mean[1]
    imgs[..., 2] -= mean[2]
    
    #print imgs
    
    net = squeezeNet(x)
    softmax = tf.nn.softmax(net)
    
    vs = tf.trainable_variables()
    # load hdf5 weights into TF tensor
    for v in vs:
        print v
        name = str(v.name).split(':')[0]
        print name
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(softmax, 1))
    
    fidLog = open(FLAGS.log_path, 'w')
    
    accuracy = 0.0
    idx = 0
    for i in xrange(int(FLAGS.batch_size/5)):
        inputImgs = imgs[idx:idx+5, :, :, :]
        
        for j in range(5):
            imarr = inputImgs[j, :, :, :]
            #imarr = imarr[..., ::-1]
            #imarr[..., 0] += mean[0]
            #imarr[..., 1] += mean[1]
            #imarr[..., 2] += mean[2]
            imarr = np.array(imarr, dtype=np.uint8)
            imshow = Image.fromarray(imarr)
            imshow.save(("./writeOut/im%05d.jpeg"%(int(i*5+j))),"jpeg")
            
        
        preLabs = labs[idx:idx+5]
        preResult = sess.run(predictions, feed_dict={x: inputImgs})
        
        print('[%s / %s]' % (i, int(FLAGS.batch_size/5)))
        fidLog.write('[%s / %s]\n' % (i+1, int(FLAGS.batch_size/5)))
        fidLog.flush()
        #print('Net predict labels: %s' % preResult)
        #print('Image Labels: %s' % labs[i:i+5])
        for j in range(5):
            if preResult[j] == preLabs[j]:
                accuracy += 1.0
        
        idx += 5
    
    #preResult = sess.run(predictions, feed_dict={x: imgs})
    #print('Net predict labels: %s' % preResult)
    #print preResult.shape
    
    accuracy = accuracy/float(FLAGS.batch_size)
    runtime = time.time()-start_time_simu
    
    # print statistics
    #print('accuracy= %s' % accuracy)
    fidLog.write('accuracy= %s\n' % accuracy)
    #print('error_list= %s' % errorList)
    #print('Runtime: %f sec'%runtime)
    fidLog.write('Runtime: %f sec\n'%runtime)
    fidLog.close()
    
    # Stop the threads
    #coord.request_stop()
    # Wait for threads to stop
    #coord.join(threads)