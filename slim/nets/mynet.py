# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:19:20 2018

@author: lynden
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from Quantize import QConv
from Quantize import QFullyConnect

slim = tf.contrib.slim

####################
### Quantizer
#intr_quantizer = None # Quantizers.FixedPointQuantizer(8,4)
#extr_quantizer = None # Quantizers.NoQuantizer()
#quantizer = None
####################


def mynet_arg_scope(weight_decay=0.0):
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      biases_initializer=tf.constant_initializer(0.1),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1)) as sc:
    return sc


def mynet(images, num_classes=10, is_training=False, reuse=None,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='MyNet',
          conv2d=slim.conv2d, 
          max_pool2d=slim.max_pool2d, 
          fully_connected = slim.fully_connected):
    """ mynet test
    """
    
    end_points = {}
    with tf.variable_scope(scope, 'MyNet', [images, num_classes]):
        net = conv2d(images, 32, [5, 5], scope='conv1')
        net = max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.flatten(net)
        end_points['Flatten'] = net
        logits = fully_connected(net, num_classes, activation_fn=None,
                                      scope='fc1')
    
    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        
    return logits, end_points

mynet.default_image_size = 28