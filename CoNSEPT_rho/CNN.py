"""
@author: Payam Dibaeinia
"""
"""
Every change from Run13 implementation was marked by #changed
"""



import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import tensorflow.keras.layers as layers

class CNN_fc(Model):
    """
    This CNN model uses a fully connected layer at the end to output expression
    """
    def __init__(self, dropout_rate, poolSize_bind, convSize_coop, stride_coop, coopAct, fcConvChan_coop, outAct):
        """
        coopAct: string
        outAct: string
        fcConvChan_coop: a list containing the number of filters (channels) for each convolutional layer after the very first coop layer
        """
        super(CNN_fc, self).__init__()
        #TODO: Is it possible to use tf.Sequential here?
        #self.bn_bind = layers.BatchNormalization()
        self.bn_bind = layers.LayerNormalization()
        self.maxPool_bind = layers.MaxPool2D(pool_size = poolSize_bind, strides = poolSize_bind)

        # after stacking, expand the last dim
        # Input to below conv has size: #batch * L * 2 * 1 (last dimension comes from expanding)
        self.conv11_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.conv22_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.conv33_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.conv12_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.conv13_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.conv23_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.coopAct = layers.Activation(coopAct)

        # output from each layer above has size : #batch * L * 1 * 1
        # Concat them to get #batch * L * 1 * 6

        self.fcConv_coop = []
        for nFilters in fcConvChan_coop:
            # We could use Conv1D, instead we used Conv2D but will squeeze once at the end
            #self.fcConv_coop.append(layers.LayerNormalization()) #changed
            self.fcConv_coop.append(layers.Conv2D(filters = nFilters, kernel_size = (3,1), strides = 1, use_bias = True, kernel_initializer='glorot_normal'))
            #self.fcConv_coop.append(layers.LayerNormalization())
            self.fcConv_coop.append(layers.Activation(coopAct))
            self.fcConv_coop.append(layers.LayerNormalization())

        self.drop_coop = layers.Dropout(rate = dropout_rate)
        self.maxPool_Coop = layers.MaxPool2D(pool_size = (3,1), strides = (3,1))
        self.flatten = layers.Flatten() #Changed from the very original implementation
        self.fc = layers.Dense(1, kernel_initializer='glorot_normal')
        self.outAct = layers.Activation(outAct)



    def call(self, inputs, training = True):
        seq, conc = inputs

        seq = self.bn_bind(seq, training = training)
        seq = self.maxPool_bind(seq)
        seq = tf.squeeze(seq, axis = 2)

        conc = tf.tile(conc, [1,seq.shape[1]])
        conc = tf.reshape(conc, (-1, seq.shape[1], 3))
        seq = tf.multiply(seq, conc)

        c11 = tf.stack((seq[:,:,0], seq[:,:,0]), axis = 2)
        c11 = tf.expand_dims(c11, -1)

        c22 = tf.stack((seq[:,:,1], seq[:,:,1]), axis = 2)
        c22 = tf.expand_dims(c22, -1)

        c33 = tf.stack((seq[:,:,2], seq[:,:,2]), axis = 2)
        c33 = tf.expand_dims(c33, -1)

        c12 = tf.stack((seq[:,:,0], seq[:,:,1]), axis = 2)
        c12 = tf.expand_dims(c12, -1)

        c13 = tf.stack((seq[:,:,0], seq[:,:,2]), axis = 2)
        c13 = tf.expand_dims(c13, -1)

        c23 = tf.stack((seq[:,:,1], seq[:,:,2]), axis = 2)
        c23 = tf.expand_dims(c23, -1)

        c11 = self.conv11_coop(c11)
        c22 = self.conv22_coop(c22)
        c33 = self.conv33_coop(c33)
        c12 = self.conv12_coop(c12)
        c13 = self.conv13_coop(c13)
        c23 = self.conv23_coop(c23)

        coop = tf.concat([c11,c22,c33,c12,c13,c23], axis = -1)
        coop = self.coopAct(coop)

        for coop_layer in self.fcConv_coop:
            #if isinstance(coop_layer, layers.BatchNormalization):
            if isinstance(coop_layer, layers.LayerNormalization):
                coop = coop_layer(coop, training = training)
            else:
                coop = coop_layer(coop)

        coop = self.drop_coop(coop, training = training)
        if coop.shape[1] > 2:
            coop = self.maxPool_Coop(coop)
        coop = tf.squeeze(coop, axis = 2)
        coop = self.flatten(coop)
        ret = self.fc(coop)
        ret = self.outAct(ret)

        if self.outAct == 'tanh':
            ret = 0.5 * (ret + 1)

        return ret
