import tensorflow as tf
import numpy as np
import cv2
import os
from stn import spatial_transformer_network as transformer
from tensorflow.keras import Model

image_height, image_width  = 24, 94
list_of_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
                "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
batch_size = 16

#LocNet Transformer Network

def small_basic_block(input, filters):
    conv_1 = tf.keras.layers.Conv2D(filters/4, kernel_size = (1,1), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(input)
    normalization_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_2 = tf.keras.layers.Conv2D(filters/4, kernel_size = (3,1), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(normalization_1)
    normalization_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_3 = tf.keras.layers.Conv2D(filters/4, kernel_size = (1,3), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(normalization_2)
    normalization_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_4 = tf.keras.layers.Conv2D(filters, kernel_size = (1,1), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(normalization_3)
    return tf.keras.layers.BatchNormalization()(conv_4)

class TransformerNet(object):
    def __init__(self, batch_size, image_height, image_width):
        input = tf.keras.Input(shape=(image_height, image_width, 3), batch_size=batch_size) # 3 is number of channels, we're taking in RGB
        #Channel 1
        avg_pool = tf.keras.layers.AveragePooling2D(pool_size = (3, 3), strides=(2, 2), padding = "same", data_format = "channels_last")(input)
        conv_1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), strides=(3,3), padding = "same", activation = "relu", data_format="channels_last")(avg_pool)
        normalization_1 = tf.keras.layers.BatchNormalization()(conv_1)

        #Channel 2
        conv_2 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(5, 5), strides=(5, 5), padding = "same", activation = "relu", data_format="channels_last")(input)
        normalization_2 = tf.keras.layers.BatchNormalization()(conv_2)

        #Merge channels
        concat = tf.keras.layers.concatenate(inputs = [normalization_1, normalization_2], axis = -1) #Channel wise concat, as channel is last dim
        dropout = tf.keras.layers.Dropout(concat, rate=0.5)
        dense_1 = tf.keras.layers.Dense(units = 32, activation = "tanh")(dropout)
        output = tf.keras.layers.Dense(units = 6, activation = "tanh")(dense_1)

        out_image = transformer(input, output)

        return out_image

class LPRNet(object):
    def __init__(self, input, batch_size, image_height, image_width, class_number):
        conv_1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", activation = "relu", data_format="channels_last")(input)
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
        return tf.keras.layers.BatchNormalization()(conv_3)




model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
