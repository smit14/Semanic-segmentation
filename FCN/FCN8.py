import os
import numpy as np
import tensorflow as tf
import cv2
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from caffe_classes import class_names
from vgg16 import model8
from Data_generator import datagenerator
import sys

#batch size and learning rate
batch_size = 2
learning_rate = .001

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


if __name__ == '__main__':

	reset_val = int(sys.argv[1])
	tf.reset_default_graph()

	tf_x = tf.placeholder(tf.float32,[None,224,224,3])
	tf_y = tf.placeholder(tf.int32,[None,224,224,1])

	vgg = model8(tf_x,batch_size)
	output = vgg.output

	data = datagenerator(batch_size)
		
	loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output[0],labels=tf.squeeze(tf_y, squeeze_dims=[3]),name="entropy")))

	trainable_var = tf.trainable_variables()
	train_op = train(loss, trainable_var)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	if reset_val == 0: 
		saver.restore(sess, "~/model8.ckpt")

	for epoch in range(30):

		for itr in range(160):

			train_batch,train_label_batch = data.get_train_batch()
			 
			sess.run(train_op,feed_dict={tf_x:train_batch, tf_y: train_label_batch.astype(int)})

			if itr%10 ==0:
				val_batch,val_label_batch = data.get_val_batch()
				print('iterations: %d , train_loss: %f , validation_loss: %f'%(itr,
					sess.run(loss,feed_dict={tf_x:train_batch, tf_y: train_label_batch.astype(int)}),
					sess.run(loss,feed_dict={tf_x:val_batch, tf_y: val_label_batch.astype(int)}) ))

	save_path = saver.save(sess, "~/model8.ckpt")




