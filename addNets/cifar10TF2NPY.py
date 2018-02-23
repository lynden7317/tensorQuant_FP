# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:15:02 2018

@author: lynden
partition cifar10 test dataset into .npy format
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import tensorflow as tf

from datasets import dataset_factory

slim = tf.contrib.slim

data_path   = '/runtmp3/lynden/imageDataset/cifar10/'
batch_size = 10000
mini_batch = 1000

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

images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=5*batch_size)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    imgs, labs = sess.run([images, labels])
    print('type of labs: %s' % type(labs))
    print('type of imgs: %s' % type(imgs))
    
    #np.save('cifaImg_10', img)
    #np.save('cifaImgLab_10', lab)
    labsList = np.split(labs, batch_size/mini_batch)
    imgsList = np.split(imgs, batch_size/mini_batch)
    for i in xrange(int(batch_size/mini_batch)):
        np.save('img_cifar10_'+str(i)+'.npy', imgsList[i])
        np.save('lab_cifar10_'+str(i)+'.npy', labsList[i])
    
    # Stop the threads
    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)