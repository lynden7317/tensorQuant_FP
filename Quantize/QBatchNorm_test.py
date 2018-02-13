import tensorflow as tf
import numpy as np
import sys
import Quantizers
import QBatchNorm as QBN

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

input_width = input_height = 28
batch_size = 10
input_channels = 8
fixed_size = 8
fixed_prec = 4
testdata_scale=10

inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale//1

inputs = tf.constant(inputs_vals,dtype=tf.float64)

means, variances = nn.moments(inputs, [0,1,2,3])

quantizer = Quantizers.NoQuantizer()
output = QBN.qbatch_normalization(inputs, means, variances, None, None, 0.0001, quantizer)
gold_output = nn.batch_normalization(inputs, means, variances, None, None, 0.0001)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  #print(sess.run(output))
  #print('------------')
  #print(sess.run(gold_output))
  print( 'mean: %f'%(sess.run(means)) )
  print( 'variance: %f'%(sess.run(variances)) )
  pass

failed=False
for i in range(len(result)):
    if result[i] != gold_result[i]:
        failed = True
        break

print('QBatchNorm test:')
if failed:
    print('---failed!---')
else:
    print('+++passed!+++')


