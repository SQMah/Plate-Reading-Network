import tensorflow as tf
import numpy as np
import cv2
import os
from stn import spatial_transformer_network as transformer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling3D, MaxPooling3D, concatenate, Dropout, Dense, ZeroPadding2D

image_height, image_width  = 24, 94
list_of_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
                "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
batch_size = 5

#LocNet Transformer Network
def small_basic_block(input, filters):
    x = Conv2D(filters/4, kernel_size = 1, strides = 1, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding = (1,0))(x)
    x = Conv2D(filters/4, kernel_size = (3,1), strides = 1, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding = (0,1))(x)
    x = Conv2D(filters/4, kernel_size = (1,3), strides = 1, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size = 1, strides = 1, activation = "relu")(x)
    return BatchNormalization()(x)

class TransformerNet(object):
    def __init__(self, batch_size, image_height, image_width):
        input = Input(shape=(image_height, image_width, 3), batch_size = batch_size) # 3 is number of channels, we're taking in RGB
        #Channel 1
        x = Conv2D(filters = 32, kernel_size=3)(input)
        x = MaxPooling3D(pool_size = (3, 3, 1), strides = (2, 2, 1))(x)
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
        keep_1 = BatchNormalization()(x)

        x = MaxPooling3D(pool_size = (3, 3, 1), strides = (1, 1, 1))(keep_1)
        keep_2 = small_basic_block(x, filters = 128)

        x = MaxPooling3D(pool_size = (3, 3, 1), strides = (1, 2, 2))(keep_2)
        x = small_basic_block(x, filters = 256)

        keep_3 = small_basic_block(x, filters = 256)
        x = MaxPooling3D(pool_size = (3, 3, 1), strides = (1, 2, 4))(keep_3)
        x = Dropout(x, rate=0.5)
        x = Conv2D(filters = 256, kernel_size = (1,4), strides = 1)(x)
        x = BatchNormalization()(x)
        x = Dropout(x, rate = 0.5)
        x = Conv2D(filters = class_number, kernel_size = (13,1), strides = 1, activation = "relu")(x)

        keep_4 = BatchNormalization()(x)

        #Global context embedding
        keep_1 = AveragePooling3D(pool_size = (5, 5, 1), strides=(5, 5, 1))(keep_1)
        keep_2 = AveragePooling3D(pool_size = (5, 5, 1), strides=(5, 5, 1))(keep_2)
        keep_3 = AveragePooling3D(pool_size = (4, 10, 1), strides=(4, 2, 1))(keep_3)

        
