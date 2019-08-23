import tensorflow as tf
import numpy as np


#---------------------------------------------------------------------- ___   ___  -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------    |     | -------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------- ___|  ___| -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------    |  |    -------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------- ___|  |___ -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------            -------------------------------------------------------------------------------------------------------------------
class model32:
    def __init__(self,x,batch_size,vgg16_npy_path='vgg16.npy'):
        
        self.batch_size = batch_size
        self.X = x
        #self.num_classes = num_Classes
        self.data_dict = np.load('vgg16.npy', encoding='latin1').item()
        
        #Input Image - [224 224 3]
        conv1_1 =  self.conv(x,3,3,64,1,1,name='conv1_1')             # [224 224 64]
        conv1_2 = self.conv(conv1_1,3,3,64,1,1,name='conv1_2')        # [224 224 64]
        pool1 = self.max_pool(conv1_2,2,2,2,2,name='pool1')           # [112 112 64]
        
        # [112 112 64]        
        conv2_1 = self.conv(pool1,3,3,128,1,1,name='conv2_1')         # [112 112 128]
        conv2_2 = self.conv(conv2_1,3,3,128,1,1,name='conv2_2')       # [112 112 128]
        pool2 = self.max_pool(conv2_2,2,2,2,2,name='pool2')           # [64 64 128]
        
        
        conv3_1 = self.conv(pool2,3,3,256,1,1,name='conv3_1')
        conv3_2 = self.conv(conv3_1,3,3,256,1,1,name='conv3_2')
        conv3_3 = self.conv(conv3_2,3,3,256,1,1,name='conv3_3')
        pool3 = self.max_pool(conv3_3,2,2,2,2,name='pool3')
        
        conv4_1 = self.conv(pool3,3,3,512,1,1,name='conv4_1')
        conv4_2 = self.conv(conv4_1,3,3,512,1,1,name='conv4_2')
        conv4_3 = self.conv(conv4_2,3,3,512,1,1,name='conv4_3')
        pool4 = self.max_pool(conv4_3,2,2,2,2,name='pool4')
        
        conv5_1 = self.conv(pool4,3,3,512,1,1,name='conv5_1')
        conv5_2 = self.conv(conv5_1,3,3,512,1,1,name='conv5_2')
        conv5_3 = self.conv(conv5_2,3,3,512,1,1,name='conv5_3')
        pool5 = self.max_pool(conv5_3,2,2,2,2,name='pool5')
        
        score5 = self.conv1(pool5,1,1,12,1,1,name='score5')
        self.output32 = self.upsample(score5,32,name='output32')
        self.output = [self.output32,self.X]
        
    def conv(self,bottom,fx,fy,fn,px,py,name=None):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
                convolve = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
                lout = tf.nn.relu(tf.nn.bias_add(convolve, self.data_dict[name][1]))
                return lout
        
    def fc(self,x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            data = self.data_dict[name]
            weights = tf.convert_to_tensor(data[0])

            biases = tf.convert_to_tensor(data[1])
            
            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def max_pool(self,x,filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)
    
    
    
    def conv1(self,bottom,filter_height,filter_width,num_filters,px,py,name):
        
        input_channels = int(bottom.get_shape()[-1])
        
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')

        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels,
                                                    num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            
        conv = convolve(bottom, weights)
        
        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu
    
    def upsample(self,bottom,stride,name):
        
        input_channels = int(bottom.get_shape()[-1])
       
        channels = (bottom.get_shape()[0])
        height = int(bottom.get_shape()[1])
        width = int(bottom.get_shape()[2])
        
        
        new_height = stride*height
        new_width = stride*width
        
        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[2*stride,
                                                    2*stride,
                                                    input_channels,
                                                    input_channels])
        
        layer = tf.nn.conv2d_transpose(bottom,weights,output_shape=[self.batch_size,new_height,new_width,input_channels],strides=[1,stride,stride,1])
        return layer



#------------------------------------------------------------ _    ___  -------------------------------------------------------------------------------------------------------
#------------------------------------------------------------  |  |     -------------------------------------------------------------------------------------------------------
#------------------------------------------------------------  |  |___  -------------------------------------------------------------------------------------------------------
#------------------------------------------------------------ _|_ |___| -------------------------------------------------------------------------------------------------------
#------------------------------------------------------------           -------------------------------------------------------------------------------------------------------
class model16:
    def __init__(self,x,batch_size,vgg16_npy_path='vgg16.npy'):
        
        self.batch_size = batch_size
        self.X = x
        #self.num_classes = num_Classes
        self.data_dict = np.load('vgg16.npy', encoding='latin1').item()
        
        #Input Image - [224 224 3]
        conv1_1 =  self.conv(x,3,3,64,1,1,name='conv1_1')             # [224 224 64]
        conv1_2 = self.conv(conv1_1,3,3,64,1,1,name='conv1_2')        # [224 224 64]
        pool1 = self.max_pool(conv1_2,2,2,2,2,name='pool1')           # [112 112 64]
        
        # [112 112 64]        
        conv2_1 = self.conv(pool1,3,3,128,1,1,name='conv2_1')         # [112 112 128]
        conv2_2 = self.conv(conv2_1,3,3,128,1,1,name='conv2_2')       # [112 112 128]
        pool2 = self.max_pool(conv2_2,2,2,2,2,name='pool2')           # [64 64 128]
        
        
        conv3_1 = self.conv(pool2,3,3,256,1,1,name='conv3_1')
        conv3_2 = self.conv(conv3_1,3,3,256,1,1,name='conv3_2')
        conv3_3 = self.conv(conv3_2,3,3,256,1,1,name='conv3_3')
        pool3 = self.max_pool(conv3_3,2,2,2,2,name='pool3')
        
        conv4_1 = self.conv(pool3,3,3,512,1,1,name='conv4_1')
        conv4_2 = self.conv(conv4_1,3,3,512,1,1,name='conv4_2')
        conv4_3 = self.conv(conv4_2,3,3,512,1,1,name='conv4_3')
        pool4 = self.max_pool(conv4_3,2,2,2,2,name='pool4')
        
        conv5_1 = self.conv(pool4,3,3,512,1,1,name='conv5_1')
        conv5_2 = self.conv(conv5_1,3,3,512,1,1,name='conv5_2')
        conv5_3 = self.conv(conv5_2,3,3,512,1,1,name='conv5_3')
        pool5 = self.max_pool(conv5_3,2,2,2,2,name='pool5')
        
        score4 = self.conv1(pool4,1,1,12,1,1,name='score4')
        mid_score5 = self.conv1(pool5,1,1,12,1,1,name='mid_score5')
        
        score5 = self.upsample(mid_score5,2,name='score5');
        mid_output16 = tf.add(score4, score5);
        
        self.output16 = self.upsample(mid_output16,16,name='output16')
        
        self.output32 = self.upsample(score5,32,name='output32')
        self.output = [self.output16,self.X]
        
    def conv(self,bottom,fx,fy,fn,px,py,name=None):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
                convolve = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
                lout = tf.nn.relu(tf.nn.bias_add(convolve, self.data_dict[name][1]))
                return lout
        
    def fc(self,x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            data = self.data_dict[name]
            weights = tf.convert_to_tensor(data[0])

            biases = tf.convert_to_tensor(data[1])
            
            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def max_pool(self,x,filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)
    
    
    
    def conv1(self,bottom,filter_height,filter_width,num_filters,px,py,name):
        
        input_channels = int(bottom.get_shape()[-1])
        
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')

        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels,
                                                    num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            
        conv = convolve(bottom, weights)
        
        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu
    
    def upsample(self,bottom,stride,name):
        
        input_channels = int(bottom.get_shape()[-1])
       
        channels = (bottom.get_shape()[0])
        height = int(bottom.get_shape()[1])
        width = int(bottom.get_shape()[2])
        
        
        new_height = stride*height
        new_width = stride*width
        
        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[2*stride,
                                                    2*stride,
                                                    input_channels,
                                                    input_channels])
        
        layer = tf.nn.conv2d_transpose(bottom,weights,output_shape=[self.batch_size,new_height,new_width,input_channels],strides=[1,stride,stride,1])
        return layer


#----------------------------------------------------------------------   ______  -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------  |      | -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------  |______| -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------  |      | -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------  |______| -------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------           -------------------------------------------------------------------------------------------------------------------

class model8:
    def __init__(self,x,batch_size,vgg16_npy_path='vgg16.npy'):
        
        self.batch_size = batch_size
        self.X = x
        #self.num_classes = num_Classes
        self.data_dict = np.load('vgg16.npy', encoding='latin1').item()
        
        #Input Image - [224 224 3]
        conv1_1 =  self.conv(x,3,3,64,1,1,name='conv1_1')             # [224 224 64]
        conv1_2 = self.conv(conv1_1,3,3,64,1,1,name='conv1_2')        # [224 224 64]
        pool1 = self.max_pool(conv1_2,2,2,2,2,name='pool1')           # [112 112 64]
        
        # [112 112 64]        
        conv2_1 = self.conv(pool1,3,3,128,1,1,name='conv2_1')         # [112 112 128]
        conv2_2 = self.conv(conv2_1,3,3,128,1,1,name='conv2_2')       # [112 112 128]
        pool2 = self.max_pool(conv2_2,2,2,2,2,name='pool2')           # [64 64 128]
        
        
        conv3_1 = self.conv(pool2,3,3,256,1,1,name='conv3_1')
        conv3_2 = self.conv(conv3_1,3,3,256,1,1,name='conv3_2')
        conv3_3 = self.conv(conv3_2,3,3,256,1,1,name='conv3_3')
        pool3 = self.max_pool(conv3_3,2,2,2,2,name='pool3')
        
        conv4_1 = self.conv(pool3,3,3,512,1,1,name='conv4_1')
        conv4_2 = self.conv(conv4_1,3,3,512,1,1,name='conv4_2')
        conv4_3 = self.conv(conv4_2,3,3,512,1,1,name='conv4_3')
        pool4 = self.max_pool(conv4_3,2,2,2,2,name='pool4')
        
        conv5_1 = self.conv(pool4,3,3,512,1,1,name='conv5_1')
        conv5_2 = self.conv(conv5_1,3,3,512,1,1,name='conv5_2')
        conv5_3 = self.conv(conv5_2,3,3,512,1,1,name='conv5_3')
        pool5 = self.max_pool(conv5_3,2,2,2,2,name='pool5')
        
        score3 = self.conv1(pool3,1,1,12,1,1,name='score3')
        mid_score4 = self.conv1(pool4,1,1,12,1,1,name='mid_score4')
        mid_score5 = self.conv1(pool5,1,1,12,1,1,name='mid_score5')
        
        score4 = self.upsample(mid_score4,2,name='score4');
        score5 = self.upsample(mid_score5,4,name='score5');
        
        mid_output8 = tf.add(tf.add(score3,score4),score5);
        
        self.output8 = self.upsample(mid_output8,8,name='output8')
        
        self.output = [self.output8,self.X]
        
    def conv(self,bottom,fx,fy,fn,px,py,name=None):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
                convolve = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
                lout = tf.nn.relu(tf.nn.bias_add(convolve, self.data_dict[name][1]))
                return lout
        
    def fc(self,x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            data = self.data_dict[name]
            weights = tf.convert_to_tensor(data[0])

            biases = tf.convert_to_tensor(data[1])
            
            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def max_pool(self,x,filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)
    
    
    
    def conv1(self,bottom,filter_height,filter_width,num_filters,px,py,name):
        
        input_channels = int(bottom.get_shape()[-1])
        
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')

        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels,
                                                    num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            
        conv = convolve(bottom, weights)
        
        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu
    
    def upsample(self,bottom,stride,name):
        
        input_channels = int(bottom.get_shape()[-1])
       
        channels = (bottom.get_shape()[0])
        height = int(bottom.get_shape()[1])
        width = int(bottom.get_shape()[2])
        
        
        new_height = stride*height
        new_width = stride*width
        
        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[2*stride,
                                                    2*stride,
                                                    input_channels,
                                                    input_channels])
        
        layer = tf.nn.conv2d_transpose(bottom,weights,output_shape=[self.batch_size,new_height,new_width,input_channels],strides=[1,stride,stride,1])
        return layer
