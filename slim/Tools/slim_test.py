import re
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

model = '../train_log/lenet/model.ckpt-120000.meta'
train_dir = '../train_log/lenet'

def main(_):
  sess = tf.Session()
  new_saver = tf.train.import_meta_graph(model)
  print(tf.get_default_graph().get_all_collection_keys())
  
  for v in tf.get_default_graph().get_collection("model_variables"):
    print(v)
    print(v.name)
  
  new_saver.restore(sess, tf.train.latest_checkpoint(train_dir))
  print(sess.run('LeNet/fc3/biases:0'))
  print(sess.run(tf.get_default_graph().get_tensor_by_name('LeNet/fc3/biases:0')))
  conv1 = sess.run(tf.get_default_graph().get_tensor_by_name('LeNet/conv1/weights:0'))
  fc3 = sess.run(tf.get_default_graph().get_tensor_by_name('LeNet/fc3/biases:0'))

  modelDict = {}
  for v in tf.get_default_graph().get_collection("model_variables"):
    print(v.name)
    modelDict[v.name] =  sess.run(tf.get_default_graph().get_tensor_by_name(v.name))
    modelDict[v.name] = np.reshape(modelDict[v.name], [-1])
    print "model name %s, shape %s" %(v.name, modelDict[v.name].shape)
    

  print conv1
  print fc3.shape
  print type(fc3)
  print fc3[0]
  #print modelDict['LeNet/conv1/weights:0']
  #print np.reshape(conv1, [-1])
  #print conv1
  #np.savetxt('./leNet/conv1', modelDict['LeNet/conv1/weights:0'])
  for v in modelDict.keys():
    nameList = v.split(':')[0].split('/')
    saveName = './'+nameList[0]+'/'+nameList[1]+'_'+nameList[2]
    np.savetxt(saveName, modelDict[v])
    #print v


if __name__ == '__main__':
  tf.app.run()
