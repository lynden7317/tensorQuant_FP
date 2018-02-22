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
import tensorflow as tf
import h5py

import utils

from datasets import dataset_factory
from preprocessing import preprocessing_factory

from Quantize import Quantizers
from Quantize import Factories 

slim = tf.contrib.slim

data_path   = '/runtmp3/lynden/imageDataset/cifar10/'
hd5Path     = './myNets/cifar10vgg.h5'
imgsPath    = './myNets/cifaImgNormalize_10.npy'
labsPath    = './myNets/cifaImgLab_10.npy'
intr_qmap   = 'qMap_cifar10Vgg16_fixed'  #'qMap_cifar10Vgg16_fixed'
extr_qmap   = ''
weight_qmap = ''
batch_size = 10
hd5ModelDict = {}

# ==== quantization type ====
intr_q_map= utils.quantizer_map(intr_qmap)
extr_q_map= utils.quantizer_map(extr_qmap)
weight_q_map = utils.quantizer_map(weight_qmap)

Qconv2d = Factories.conv2d_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map, weight_q_map=weight_q_map)
Qfully_connected = Factories.fully_connected_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map, weight_q_map=weight_q_map)
Qmax_pool2d = Factories.max_pool2d_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map)
Qavg_pool2d = Factories.avg_pool2d_factory(
                intr_q_map=intr_q_map, extr_q_map=extr_q_map)
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
            print '** rename = ', rename
            hd5ModelDict['cifar10vgg_16/'+rename] = node.value
        else:
            split = str(name).split('/')
            rename = split[1]+'/'+split[2]
            print '** rename = ', rename
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
          batch_norm=slim.batch_norm,
          fully_connected = slim.fully_connected):
    """ cifar10 for vgg16 test
    """
    #end_points = {}
    with tf.variable_scope(scope, 'cifar10vgg_16', [images, num_classes]):
        net = conv2d(inputs, 64, [3, 3], scope='conv2d_1')
        net = batch_norm(net, scale=True, scope='batch_normalization_1')
        net = conv2d(net, 64, [3, 3], scope='conv2d_2')
        net = batch_norm(net, scale=True, scope='batch_normalization_2')
        net = max_pool2d(net, [2, 2], scope='pool1')
        
        net = conv2d(net, 128, [3, 3], scope='conv2d_3')
        net = batch_norm(net, scale=True, scope='batch_normalization_3')
        net = conv2d(net, 128, [3, 3], scope='conv2d_4')
        net = batch_norm(net, scale=True, scope='batch_normalization_4')
        net = max_pool2d(net, [2, 2], scope='pool2')
        
        net = conv2d(net, 256, [3, 3], scope='conv2d_5')
        net = batch_norm(net, scale=True, scope='batch_normalization_5')
        net = conv2d(net, 256, [3, 3], scope='conv2d_6')
        net = batch_norm(net, scale=True, scope='batch_normalization_6')
        net = conv2d(net, 256, [3, 3], scope='conv2d_7')
        net = batch_norm(net, scale=True, scope='batch_normalization_7')
        net = max_pool2d(net, [2, 2], scope='pool3')
        
        net = conv2d(net, 512, [3, 3], scope='conv2d_8')
        net = batch_norm(net, scale=True, scope='batch_normalization_8')
        net = conv2d(net, 512, [3, 3], scope='conv2d_9')
        net = batch_norm(net, scale=True, scope='batch_normalization_9')
        net = conv2d(net, 512, [3, 3], scope='conv2d_10')
        net = batch_norm(net, scale=True, scope='batch_normalization_10')
        net = max_pool2d(net, [2, 2], scope='pool4')
        
        net = conv2d(net, 512, [3, 3], scope='conv2d_11')
        net = batch_norm(net, scale=True, scope='batch_normalization_11')
        net = conv2d(net, 512, [3, 3], scope='conv2d_12')
        net = batch_norm(net, scale=True, scope='batch_normalization_12')
        net = conv2d(net, 512, [3, 3], scope='conv2d_13')
        net = batch_norm(net, scale=True, scope='batch_normalization_13')
        net = max_pool2d(net, [2, 2], scope='pool5')
        
        net = slim.flatten(net)
        net = fully_connected(net, 512, scope='dense_1')
        net = fully_connected(net, num_classes, scope='dense_2')
    
    return net

# ==== prepare dataset ====
dataset = dataset_factory.get_dataset(
        'cifar10', 'test', data_path)

print('dataset= %s' % dataset)

provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=8 * batch_size,
        common_queue_min=batch_size*4)

[image, label] = provider.get(['image', 'label'])

x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])

# select the preprocessing function
#preprocessing_name = 'vgg_16' #'cifarnet'
#image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#        preprocessing_name,
#        is_training=False)
#eval_image_size = 32
#image = image_preprocessing_fn(image, eval_image_size, eval_image_size)


# === read hdf5 ====
f5 = h5py.File(hd5Path, 'r')

print("Keys: %s" % f5.keys())
f5.visititems(hefVisitor_func)
#print hd5ModelDict.keys()


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=5 * batch_size)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    imgs = np.load(imgsPath)
    labs = np.load(labsPath)
        
    #img, lab = sess.run([images, labels])
    #print('lab: %s' % lab)
    #print('img: %s' % img)
    #np.save('cifaImg_10', img)
    #np.save('cifaImgLab_10', lab)
    
    #net = cifar10vgg_16(images)
    net = cifar10vgg_16(x, conv2d=Qconv2d, 
                           max_pool2d=Qmax_pool2d, 
                           fully_connected=Qfully_connected)
    softmax = tf.nn.softmax(net)
    
    vs = tf.trainable_variables()
    # load hdf5 weights into TF tensor
    for v in vs:
        print v
        name = str(v.name).split(':')[0]
        print name
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    moving_means = tf.contrib.framework.get_variables_by_suffix('moving_mean')
    moving_variances = tf.contrib.framework.get_variables_by_suffix('moving_variance')
    #variance_is_equal = tf.reduce_all(tf.equal(*moving_variances))
    #print([v.name for v in moving_variances])
    for v in moving_means:
        print v
        name = str(v.name).split(':')[0]
        print name
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    for v in moving_variances:
        print v
        name = str(v.name).split(':')[0]
        print name
        v.load(hd5ModelDict[name], sess)
        assert np.all(sess.run(v) == hd5ModelDict[name])
    
    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(softmax, 1))
    labels = tf.squeeze(labels)
    #preResult = sess.run([predictions, labels])
    preResult = sess.run(predictions, feed_dict={x: imgs})
    
    print('Net predict labels: %s' % preResult)
    print('Image Labels: %s' % labs)
    
    accuracy = 0.0
    for i in range(batch_size):
        if preResult[i] == labs[i]:
            accuracy += 1.0
    
    accuracy = accuracy/float(batch_size)
    print('[accuracy]= %s' % accuracy)
    
  
    # Stop the threads
    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)