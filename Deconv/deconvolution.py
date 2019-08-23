import os
import numpy as np
import tensorflow as tf
import cv2
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from caffe_classes import class_names
from deconvmodel import model
from datagenerator import datagenerator
import sys

#batch size and learning rate
batch_size = 2
learning_rate = .001

#train function
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)



if __name__ == '__main__':

	reset_val = int(sys.argv[1])
	tf.reset_default_graph()

	tf_x = tf.placeholder(tf.float32,[batch_size,7,7,512])
	tf_ind5 = tf.placeholder(tf.float32,[batch_size,14,14,512])
	tf_ind4 = tf.placeholder(tf.float32,[batch_size,28,28,512])
	tf_ind3 = tf.placeholder(tf.float32,[batch_size,56,56,256])
	tf_ind2 = tf.placeholder(tf.float32,[batch_size,112,112,128])
	tf_ind1 = tf.placeholder(tf.float32,[batch_size,224,224,64])

	vgg = model(tf_x,tf_ind5,tf_ind4,tf_ind3,tf_ind2,tf_ind1)
	output = vgg.final

	data = datagenerator(batch_size)

	tf_y = tf.placeholder(tf.int32,[None,224,224,1])

	loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=tf.squeeze(tf_y, squeeze_dims=[3]),name="entropy")))
	trainable_var = tf.trainable_variables()
	train_op = train(loss, trainable_var)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	if reset_val == 0: 
		saver.restore(sess, "~/modeldeconv1.ckpt")

	for epoch in range(30):
		print('----------------------epoch: %d -------------------------------'%epoch)
    
		for itr in range(160):

			imgs,labels,ind5,ind4,ind3,ind2,ind1 = data.get_train_batch()


			sess.run(train_op,feed_dict={tf_x:imgs,tf_ind5:ind5,tf_ind4:ind4,tf_ind3:ind3,tf_ind2:ind2,tf_ind1:ind1,tf_y: labels.astype(int)})

			if itr%10 ==0:

				v_imgs,v_labels,v_ind5,v_ind4,v_ind3,v_ind2,v_ind1 = data.get_val_batch()	

				print('iterations: %d , train_loss: %f , validation_loss: %f'%(itr,
				sess.run(loss,feed_dict={tf_x:imgs,tf_ind5:ind5,tf_ind4:ind4,tf_ind3:ind3,tf_ind2:ind2,tf_ind1:ind1,tf_y: labels.astype(int)}),
				sess.run(loss,feed_dict={tf_x:v_imgs,tf_ind5:v_ind5,tf_ind4:v_ind4,tf_ind3:v_ind3,tf_ind2:v_ind2,tf_ind1:v_ind1,tf_y: v_labels.astype(int)})))

		save_path = saver.save(sess, "~/modeldeconv1.ckpt")

	save_path = saver.save(sess, "~/modeldeconv1.ckpt")





