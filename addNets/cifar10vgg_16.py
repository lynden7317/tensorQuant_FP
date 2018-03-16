# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:23:46 2018

@author: lynden
vgg_16 for cifra-10
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


tf.app.flags.DEFINE_string(
    'hdf5_path', '', 'Location of hdf5 weight file.')
tf.app.flags.DEFINE_string(
    'img_path', '', 'Location of image dataset.')
tf.app.flags.DEFINE_string(
    'lab_path', '', 'Location of label dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1000, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'log_path', '', 'Location of log.')

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

# ==== quantization type ====
intr_q_map= utils.quantizer_map(FLAGS.intr_qmap)
extr_q_map= utils.quantizer_map(FLAGS.extr_qmap)
weight_q_map = utils.quantizer_map(FLAGS.weight_qmap)


Qconv2d = Factories.conv2d_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map, weight_q_map=weight_q_map)
Qfully_connected = Factories.fully_connected_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map, weight_q_map=weight_q_map)
Qmax_pool2d = Factories.max_pool2d_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map)
Qavg_pool2d = Factories.avg_pool2d_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map)
# batch_norm quantizer
#Qbatch_norm = intr_q_map['cifar10vgg_16/batch_normalization_1']
# ===========================

# ==== hdf5 preprocessing ====
def hefVisitor_func(name, node):
    global hd5ModelDict
    if isinstance(node, h5py.Dataset):
        #print("name %s is dataset" % name)
        #print("name split %s" % str(name).split('_'))
        #print 'node shape = ', node.value.shape
        if str(name).split('_')[0] in ['dense']:
            split = str(name).split('/')
            rename = split[1]+'/'+split[2]
            #print '** rename = ', rename
            hd5ModelDict['cifar10vgg_16/'+rename] = node.value
        else:
            split = str(name).split('/')
            rename = split[1]+'/'+split[2]
            #print '** rename = ', rename
            hd5ModelDict['cifar10vgg_16/'+rename] = node.value
    else:
        pass
# ============================

# ==== Network model =========
def cifar10vgg_16(inputs, num_classes=10, is_training=False, reuse=None,
          dropout_keep_prob=0.3,
          prediction_fn=slim.softmax,
          activation_fn=tf.nn.relu,
          scope='cifar10vgg_16',
          conv2d=slim.conv2d, 
          max_pool2d=slim.max_pool2d,
          batch_norm=QBatchNorm.batch_norm,
          fully_connected = slim.fully_connected,
          quantizer=None):
    """ cifar10 for vgg16 test
    """
    #end_points = {}
    with tf.variable_scope(scope, 'cifar10vgg_16', [inputs, num_classes]):
        net = conv2d(inputs, 64, [3, 3], scope='conv2d_1')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_1',
                         quantizer=quantizer)
        net = conv2d(net, 64, [3, 3], scope='conv2d_2')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_2',
                         quantizer=quantizer)
        net = max_pool2d(net, [2, 2], scope='pool1')
        
        net = conv2d(net, 128, [3, 3], scope='conv2d_3')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_3',
                         quantizer=quantizer)
        net = conv2d(net, 128, [3, 3], scope='conv2d_4')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_4',
                         quantizer=quantizer)
        net = max_pool2d(net, [2, 2], scope='pool2')
        
        net = conv2d(net, 256, [3, 3], scope='conv2d_5')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_5',
                         quantizer=quantizer)
        net = conv2d(net, 256, [3, 3], scope='conv2d_6')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_6',
                         quantizer=quantizer)
        net = conv2d(net, 256, [3, 3], scope='conv2d_7')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_7',
                         quantizer=quantizer)
        net = max_pool2d(net, [2, 2], scope='pool3')
        
        net = conv2d(net, 512, [3, 3], scope='conv2d_8')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_8',
                         quantizer=quantizer)
        net = conv2d(net, 512, [3, 3], scope='conv2d_9')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_9',
                         quantizer=quantizer)
        net = conv2d(net, 512, [3, 3], scope='conv2d_10')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_10',
                         quantizer=quantizer)
        net = max_pool2d(net, [2, 2], scope='pool4')
        
        net = conv2d(net, 512, [3, 3], scope='conv2d_11')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_11',
                         quantizer=quantizer)
        net = conv2d(net, 512, [3, 3], scope='conv2d_12')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_12',
                         quantizer=quantizer)
        net = conv2d(net, 512, [3, 3], scope='conv2d_13')
        net = batch_norm(net, scale=True, is_training=is_training, scope='batch_normalization_13',
                         quantizer=quantizer)
        net = max_pool2d(net, [2, 2], scope='pool5')
        
        net = slim.flatten(net)
        net = fully_connected(net, 512, scope='dense_1')
        net = fully_connected(net, num_classes, scope='dense_2')
    
    return net


# the place to locate images
#x = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3])
x = tf.placeholder(tf.float32, [5, 32, 32, 3])

# === read hdf5 ====
f5 = h5py.File(FLAGS.hdf5_path, 'r')

#print("Keys: %s" % f5.keys())
f5.visititems(hefVisitor_func)
#print hd5ModelDict.keys()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    start_time_simu = time.time()
    
    imgs = np.load(FLAGS.img_path)
    labs = np.load(FLAGS.lab_path)
    # ==== image array value normalization ====
    mean = 120.707
    std = 64.15
    imgs = (imgs-mean)/(std+1e-7)
        
    
    net = cifar10vgg_16(x, conv2d=Qconv2d, 
                           max_pool2d=Qmax_pool2d, 
                           fully_connected=Qfully_connected)
    softmax = tf.nn.softmax(net)
    
    vs = tf.trainable_variables()
    # load hdf5 weights into TF tensor
    for v in vs:
        #print v
        name = str(v.name).split(':')[0]
        #print name
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    moving_means = tf.contrib.framework.get_variables_by_suffix('moving_mean')
    moving_variances = tf.contrib.framework.get_variables_by_suffix('moving_variance')
    #variance_is_equal = tf.reduce_all(tf.equal(*moving_variances))
    #print([v.name for v in moving_variances])
    for v in moving_means:
        name = str(v.name).split(':')[0]
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    for v in moving_variances:
        name = str(v.name).split(':')[0]
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(softmax, 1))

    
    fidLog = open(FLAGS.log_path, 'w')
    
    accuracy = 0.0
    idx = 0
    for i in xrange(int(FLAGS.batch_size/5)):
        inputImgs = imgs[idx:idx+5, :, :, :]
        preLabs = labs[idx:idx+5]
        preResult = sess.run(predictions, feed_dict={x: inputImgs})
        
        #print('[%s / %s]' % (i, int(FLAGS.batch_size/5)))
        fidLog.write('[%s / %s]\n' % (i+1, int(FLAGS.batch_size/5)))
        fidLog.flush()
        #print('Net predict labels: %s' % preResult)
        #print('Image Labels: %s' % labs[i:i+5])
        for j in range(5):
            if preResult[j] == preLabs[j]:
                accuracy += 1.0
        
        idx += 5
    
    '''
    # ==== no partiton batch ===
    preResult = sess.run(predictions, feed_dict={x: imgs})
    
    print('Net predict labels: %s' % preResult)
    print('Image Labels: %s' % labs)
    
    accuracy = 0.0
    errorList = []
    for i in range(FLAGS.batch_size):
        if preResult[i] == labs[i]:
            accuracy += 1.0
        else:
            errorList.append((i, [preResult[i], labs[i]]))
    # =========================
    '''
    
    accuracy = accuracy/float(FLAGS.batch_size)
    runtime = time.time()-start_time_simu
    
    # print statistics
    #print('accuracy= %s' % accuracy)
    fidLog.write('accuracy= %s\n' % accuracy)
    #print('error_list= %s' % errorList)
    #print('Runtime: %f sec'%runtime)
    fidLog.write('Runtime: %f sec\n'%runtime)
    fidLog.close()