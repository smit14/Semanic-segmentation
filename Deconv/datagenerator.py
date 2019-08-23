import numpy as np
import os
import cv2
import skimage.io
import skimage.transform
import tensorflow as tf
from vgg16 import model


class datagenerator:

	def __init__(self,batch_size):

		self.batch_size = batch_size
		self.direct = '../CamVid'
		self.train_dir = self.direct + '/train'
		self.train_label_dir = self.direct + '/trainannot'
		self.test_dir = self.direct + '/test'
		self.test_label_dir = self.direct + '/testannot'
		self.val_dir = self.direct + '/val'
		self.val_label_dir = self.direct + '/valannot'

		#training files info
		self.train_files = os.listdir(self.train_dir)
		self.train_files = sorted(self.train_files)
		self.train_num = len(self.train_files)
		self.train_imgs = np.zeros([self.train_num,224,224,3])

	    #training image info
		self.train_label_files = os.listdir(self.train_label_dir)
		self.train_label_files = sorted(self.train_label_files)
		self.train_label_num = len(self.train_files)
		self.train_label_imgs = np.zeros([self.train_label_num,224,224,1])

		#validation files info
		self.val_files = os.listdir(self.val_dir)
		self.val_files = sorted(self.val_files)
		self.val_num = len(self.val_files)
		self.val_imgs = np.zeros([self.val_num,224,224,3])

		#validation image info
		self.val_label_files = os.listdir(self.val_label_dir)
		self.val_label_files = sorted(self.val_label_files)
		self.val_label_num = len(self.val_files)
		self.val_label_imgs = np.zeros([self.val_label_num,224,224,1])

		# train images and labels loading
		for i in range(self.train_num):
		    self.train_imgs[i,:,:,:] = self.load_img(self.train_dir+'/'+self.train_files[i])
		    
		for i in range(self.train_label_num):
		    self.train_label_imgs[i,:,:,:] = self.load_img(self.train_label_dir+'/'+self.train_label_files[i],label=True)    


		# val images and labels loading
		for i in range(self.val_num):
		    self.val_imgs[i,:,:,:] = self.load_img(self.val_dir+'/'+self.val_files[i])

		for i in range(self.val_label_num):
		    self.val_label_imgs[i,:,:,:] = self.load_img(self.val_label_dir+'/'+self.val_label_files[i],label=True)

		self.reset_train()
		self.reset_val()

	def reset_train(self):
		self.arr_train = np.arange(self.train_num)
		np.random.shuffle(self.arr_train)
		self.idx_train = 0

	def reset_val(self):
		self.arr_val = np.arange(self.val_num)
		np.random.shuffle(self.arr_val)
		self.idx_val = 0


	def get_train_batch(self):
		
		if self.idx_train == (self.train_num//self.batch_size)*self.batch_size:
			self.reset_train()

		self.train_arr_imgs = [self.train_imgs[i] for i in self.arr_train[self.idx_train:self.idx_train+self.batch_size]]
		self.train_arr_labels = [self.train_label_imgs[i] for i in self.arr_train[self.idx_train:self.idx_train+self.batch_size]]

		self.train_arr_imgs = np.asarray(self.train_arr_imgs)
		self.train_arr_labels = np.asarray(self.train_arr_labels)

		self.idx_train += self.batch_size

		self.train_arr_imgs, self.train_idx5,self.train_idx4,self.train_idx3,self.train_idx2,self.train_idx1 = self.get_indexes(self.train_arr_imgs)

		return self.train_arr_imgs,self.train_arr_labels, self.train_idx5,self.train_idx4,self.train_idx3,self.train_idx2,self.train_idx1

	def get_val_batch(self):
		if self.idx_val == (self.val_num//self.batch_size)*self.batch_size:
			self.reset_val()

		self.val_arr_imgs = [self.val_imgs[i] for i in self.arr_val[self.idx_val:self.idx_val+self.batch_size]]
		self.val_arr_labels = [self.val_label_imgs[i] for i in self.arr_val[self.idx_val:self.idx_val+self.batch_size]]

		self.val_arr_imgs = np.asarray(self.val_arr_imgs)
		self.val_arr_labels = np.asarray(self.val_arr_labels)

		self.idx_val += self.batch_size

		self.val_arr_imgs, self.val_idx5,self.val_idx4,self.val_idx3,self.val_idx2,self.val_idx1 = self.get_indexes(self.val_arr_imgs)

		return self.val_arr_imgs,self.val_arr_labels, self.val_idx5,self.val_idx4,self.val_idx3,self.val_idx2,self.val_idx1

	def load_img(self,path,label=False):
		vgg_mean = [123.68, 116.779, 103.939]
		img = cv2.imread(path)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = cv2.resize(img.astype(np.float32), (224,224))
		img = img.reshape((224,224,3))
		if not label:
			img -= vgg_mean
		if label:
			return img[:,:,0:1]
		return img

	def get_indexes(self,arr):

		tf_x = tf.placeholder(tf.float32,[None,224,224,3])
		vgg = model(tf_x,self.batch_size)
		output = vgg.output
		
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		out = sess.run(output,feed_dict={tf_x:arr})
		
		ls = [out[0]]

		for i in range(len(out)-1):
			ls.append(self.get_indices(out[i+1]))
			
		return ls

	def get_indices(self,inp):	

		sz = inp.shape
		out_batch = sz[0]
		out_height = 2*sz[1]
		out_width = 2*sz[2]
		out_channels = sz[3]


		out = np.zeros([out_batch*out_width*out_height*out_channels])

		for ch in range(out_channels):
		    for i in range(sz[1]):
		        for j in range(sz[2]):
		            for k in range(sz[0]):
		                out[inp[k][i][j][ch]] = 1

		return out.reshape(out_batch,out_width,out_height,out_channels)
