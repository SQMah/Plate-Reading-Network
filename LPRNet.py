import tensorflow as tf
import numpy as np
import cv2
import os
from stn import spatial_transformer_network as transformer
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D, concatenate, Dropout, Dense, ZeroPadding2D

image_height, image_width  = 24, 94
list_of_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
                "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
batch_size = 5

#LocNet Transformer Network
def small_basic_block(input, filters):
    x = Conv2D(filters/4, kernel_size = 1, strides = 1, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding = (1,0), data_format="channels_last")(x)
    x = Conv2D(filters/4, kernel_size = (3,1), strides = 1, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding = (0,1), data_format="channels_last")(x)
    x = Conv2D(filters/4, kernel_size = (1,3), strides = 1, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size = 1, strides = 1, activation = "relu")(x)
    return BatchNormalization()(x)

class TransformerNet(object):
    def __init__(self, batch_size, image_height, image_width):
        input = Input(shape=(image_height, image_width, 3), batch_size = batch_size) # 3 is number of channels, we're taking in RGB
        #Channel 1
        x = AveragePooling2D(pool_size = 3, strides = 2)(input)
        x = Conv2D(filters = 32, kernel_size = 5, strides = 3, activation = "relu")(x)
        x = BatchNormalization()(x)

        #Channel 2
        y = Conv2D(filters = 32, kernel_size = 5, strides = 5, activation = "relu")(input)
        y = BatchNormalization()(y)

        #Merge channels
        x = concatenate(inputs = [x, y], axis = -1) #Channel wise concat, as channel is last dim
        x = Dropout(x, rate=0.5)
        x = Dense(units = 32, activation = "tanh")(x)
        x = Dense(units = 6, activation = "tanh")(x)

        out_image = transformer(input, x)

        return out_image

class LPRNet(object):
    def __init__(self, input, batch_size, image_height, image_width, class_number):
        x = Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = "relu")(input)
        normalization_1 = tf.keras.layers.BatchNormalization()(conv_1)
        pool_1 = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same", data_format="channels_last")(normalization_1)
        basic_block_1 = small_basic_block(pool_1, filters = 128)
        pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides = (2,1), padding = "same", data_format="channels_last")(basic_block_1)
        basic_block_2 = small_basic_block(pool_2, filters = 256)
        basic_block_3 = small_basic_block(basic_block_2, filters = 256)
        pool_3 = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (2,1), padding = "same", data_format="channels_last")(basic_block_3)
        dropout_1 = tf.keras.layers.Dropout(pool_3, rate=0.5)
        conv_2 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (4,1), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(dropout_1)
        normalization_2 = tf.keras.layers.BatchNormalization()(conv_2)
        dropout_2 = tf.keras.layers.Dropout(normalization_2, rate = 0.5)
        conv_3 = tf.keras.layers.Conv2D(filters = class_number, kernel_size = (1,13), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(dropout_2)
        normalization_3 = tf.keras.layers.BatchNormalization()(conv_3)

        #Global context embedding
        dense_1 = tf.keras.layers.Dense(units = class_number, activation = "relu")(normalization_3)
        concat = tf.keras.layers.concatenate(inputs = [normalization_3, dense_1], axis = -1) # Channel wise concat
        conv_4 = tf.keras.layers.Conv2D(filters = class_number, kernel_size = (1,1), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(concat)
        normalization_4 = tf.keras.layers.BatchNormalization()(conv_4)
