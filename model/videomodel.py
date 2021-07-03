import numpy as np
import pandas as pd
import itertools 
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, Input, Dense, MaxPool2D,BatchNormalization,Flatten,ZeroPadding2D,Activation,Dropout,LSTM
from keras.layers import TimeDistributed
from keras.models import Model,Sequential

def classifierModel(X_input):
  X=TimeDistributed(Conv2D(64, (3,3), strides=(2,2), name='conv1', activation='relu'))(X_input)
  X=TimeDistributed(MaxPool2D((3,3),strides=(2,2),name='maxpool2'))(X)
  X = TimeDistributed(BatchNormalization())(X)

  X = TimeDistributed(Conv2D(96,(1,1),name='conv4',activation='relu'))(X)
  X = TimeDistributed(MaxPool2D((3,3),strides=(1,1),name='max_pool3'))(X)
  X = TimeDistributed(Conv2D(208,(3,3),name='conv5',activation='relu'))(X)
  X = TimeDistributed(Conv2D(64,(1,1),name='conv6',activation='relu'))(X)

  X= TimeDistributed(Conv2D(96,(1,1),name='conv7',activation='relu'))(X)
  X= TimeDistributed(MaxPool2D((3,3),strides=(1,1),name='max_pool4'))(X)
  X = TimeDistributed(Conv2D(208, (3,3),name='conv8',activation='relu'))(X)
  X = TimeDistributed(Conv2D(64,(1,1),name='conv9',activation='relu'))(X)

  out = TimeDistributed(Flatten())(X)
  out = TimeDistributed(Dropout(0.5))(out)
  out = TimeDistributed(Dense(128,activation = 'linear'))(out)


  return out

def LSTM_model(input_shape):
  X_input = Input(shape = input_shape)
  X = classifierModel(X_input)
  X = LSTM(128)(X)
  X = Dense(8,activation='softmax')(X)
  model = Model(inputs = X_input,outputs = X)
  return model

model1 = LSTM_model((12,48,48,1))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

from keras.layers import Conv3D,MaxPool3D
def c3d_model(input_shape):
  input = Input(shape = input_shape)
  X = Conv3D(64,(1,3,3),activation = 'relu')(input)
  X = MaxPool3D((1,2,2),strides=(1,2,2))(X)
  X = Conv3D(128,(1,3,3),activation='relu')(X)
  X = MaxPool3D((1,2,2),strides=(2,2,2))(X)
  X = Conv3D(128,(1,1,1),activation='relu')(X)
  X = Conv3D(256,(1,1,1),activation = 'relu')(X)
  X = MaxPool3D((2,2,2),strides=(2,2,2))(X)
  X = Conv3D(256,(1,1,1),activation='relu')(X)
  X = Conv3D(512,(1,1,1),activation='relu')(X)
  X = MaxPool3D((2,2,2),strides=(2,2,2))(X)
  X = Conv3D(512,(1,1,1),activation='relu')(X)
  X = Conv3D(512,(1,1,1),activation='relu')(X)
  X = MaxPool3D((1,1,1),strides=(2,2,2))(X)
  X = Flatten()(X)
  X = Dense(4096)(X)
  X = Dropout(0.5)(X)
  X = Dense(4096)(X)
  X = Dropout(0.5)(X)
  out = Dense(8,activation='softmax')(X)
  model = Model(inputs = input,outputs = out)
  return model

c3d = c3d_model((12,48,48,1))
c3d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
