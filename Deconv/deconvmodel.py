import numpy as np
import tensorflow as tf

class model:
    def __init__(self,x,ind5,ind4,ind3,ind2,ind1):
        
        # x : input -> output of pool5 with size [batch_size,7,7,512]
        # ind5 = maxpool indices of pool5, size = [batch_size,14,14,512]
        # ind4 = maxpool indices of pool5, size = [batch_size,28,28,512]
        # ind3 = maxpool indices of pool5, size = [batch_size,56,56,256]
        # ind2 = maxpool indices of pool5, size = [batch_size,112,112,128]
        # ind1 = maxpool indices of pool5, size = [batch_size,224,224,64]
        
        unpool5 = self.unpool(x,ind5)                                               # [14 14 512]
        deconv5_3 = self.upsample_layer(unpool5,3,3,512,1,1,"deconv5_3")            # [14 14 512]
        deconv5_2 = self.upsample_layer(deconv5_3,3,3,512,1,1,"deconv5_2")          # [14 14 512]
        deconv5_1 = self.upsample_layer(deconv5_2,3,3,512,1,1,"deconv5_1")     # [14 14 512]
        
        
        unpool4 = self.unpool(deconv5_1,ind4)                                  # [28 28 512]
        deconv4_3 = self.upsample_layer(unpool4,3,3,512,1,1,"deconv4_3")       # [28 28 512]
        deconv4_2 = self.upsample_layer(deconv4_3,3,3,512,1,1,"deconv4_2")     # [28 28 512]
        deconv4_1 = self.upsample_layer(deconv4_2,3,3,256,1,1,"deconv4_1")     # [28 28 256]
        
        
        unpool3 = self.unpool(deconv4_1,ind3)                                  # [56 56 256]
        deconv3_3 = self.upsample_layer(unpool3,3,3,256,1,1,"deconv3_3")       # [56 56 256]
        deconv3_2 = self.upsample_layer(deconv3_3,3,3,256,1,1,"deconv3_2")     # [56 56 256]
        deconv3_1 = self.upsample_layer(deconv3_2,3,3,128,1,1,"deconv3_1")     # [56 56 128]
        
        
        unpool2 = self.unpool(deconv3_1,ind2)                                  # [112 112 128]
        deconv2_2 = self.upsample_layer(unpool2,3,3,128,1,1,"deconv2_2")       # [112 112 128]
        deconv2_1 = self.upsample_layer(deconv2_2,3,3,64,1,1,"deconv2_1")      # [112 112 64]
        
        
        
        unpool1 = self.unpool(deconv2_1,ind1)                                  # [224 224 64]
        deconv1_2 = self.upsample_layer(unpool1,3,3,64,1,1,"deconv1_2")        # [224 224 64]
        deconv1_1 = self.upsample_layer(deconv1_2,3,3,3,1,1,"deconv1_1")       # [224 224 3]
        
        self.final = self.conv1(deconv1_1,3,3,12,1,1,name="final")
        
        
    # unpooling process    
    def unpool(self,bottom,indices):
        
        # bottom: inpoot layer , size: [batch_size,m,m,c]
        # indices: indices of maxpool , size:[batch_size,2m,2m,c]
        
        sz = bottom.shape
        
        wd = sz[1]         # width
        hg = sz[2]         # height 
        channels = sz[3]   # channels
        
        
        # create matriz with all 1 of size [heightt, 2*height]
        w = self.create_w(hg)
        w = tf.constant(w,dtype=tf.float32)


        
        output1 = []
        
        
        for i in range(sz[0]):
            output = []
            for c in range(channels):

                inp = bottom[i,:,:,c]

                ind = indices[i,:,:,c]


                out1 = tf.tensordot(inp,w,1)

                out2 = tf.tensordot(tf.transpose(out1),w,1)
                out = tf.multiply(tf.transpose(out2),ind)
                
                output.append(out);

            output1.append(tf.stack(output,axis=2))
            
        output2 = tf.stack(output1,axis=0)
        
        return output2
    
    def create_w(self,size):
        w = np.zeros([size,2*size])
        for i in range(size):
            for j in range(2*i,2*i+2):
                w[i,j] = 1
        return w
    
    
    def upsample_layer(self,bottom,filter_height,filter_width,num_filters,px,py,name):
        
        input_channels = int(bottom.get_shape()[-1])
        sz = bottom.shape.as_list()
        
        sz[-1] = num_filters
        output_shape = tf.stack(sz)
        
        with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    num_filters,input_channels])
        
        ct =  tf.nn.conv2d_transpose(bottom,weights,sz,[1,1,1,1],padding='SAME')
        return tf.nn.relu(ct,name=name)
    
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