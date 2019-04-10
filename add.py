#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import datetime
import os
import numpy as np
import sklearn
from sklearn.metrics import f1_score, accuracy_score, classification_report,hamming_loss
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle, gzip
import sys
np.random.seed(26)
# import sys,shutil,datetime,pickle,codecs,tempfile, gzip

# import numpy as np
# import scipy as sp
import pandas as pd
# import matplotlib as mpl
# import cv2

# import sklearn
# import skimage

# import tensorflow as tf
# import keras
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.naive_bayes import GaussianNB,MultinomialNB


# # Init_params

# In[2]:


save_dir=os.getcwd()
model_name='model_basic_nn'


# # Loading Model 

# In[3]:



if os.path.isdir(save_dir) and model_name in os.listdir(save_dir):
    model_path = os.path.join(save_dir, model_name)
    model=tensorflow.keras.models.load_model(model_path)
    print('Loading Saved Model')
else:
    print('Model Not found, please check path')


# # load DataSet

# In[5]:


#input_file=sys.argv[1]
input_file='sample.in'
df  = pd.read_csv(input_file, sep=',',header=None)
X = df[0]
print('DataSet Loaded',X.shape)


# # Convert to 52 feature vector 

# In[6]:


mapper={'A':14,'K':13,'Q':12,'J':11,'T':10}
for i in range(9,1,-1):
    mapper[str(i)]=i
#print(mapper)
#print(X.shape)
X[0].split(".")
X52=np.zeros((X.shape[0],52))
for i in range(X.shape[0]):
    x=X[i]
    splits=x.split(".")
    for j in range(4):
        for s in splits[j] :
            ind=(j*13+(mapper[s]-2))
            X52[i][ind]=1
   
            #X52[i][0]=1 
        
print('dataSet Converted to 52 dim feature vector',X52.shape)


# # Predict

# In[7]:


y_pred=model.predict(X52)
y_pred=1*(y_pred-np.mean(y_pred,axis=1).reshape(-1,1))/(np.std(y_pred,axis=1).reshape(-1,1))
y_pred=1*(y_pred>=0)


# # Writing to file

# In[11]:


#[REF] : https://stackoverflow.com/questions/3345336/save-results-to-csv-file-with-python
np.savetxt(sys.argv[2], (y_pred), delimiter=',', fmt='%s')


# In[ ]:


# #f2 = open(sys.argv[2],'w')
# f2 = open('sampletest.out','w')
# for y in y_pred:
#     f2.write( str(y)+"\n")
# f2.close()


# # Code used for training model can be seen in pds_mohit_3 
#Code for modeling my neural network

def task7(X_train,y_train,X_val,y_val,n_epochs=50):
  
  model = Sequential()
  
  m=X_train.shape[1]
  l=y_train.shape[1]
  layers=[5000]
  acts=['relu']*len(layers)
  filters=[1,1]
  config=[layers]+[acts]+[filters]
  model.add(Flatten(input_shape=X_train.shape[1:]))
  #model.add(Dense(512, activation=acts[0], input_dim=m))
  #model.add(Dropout(0.5))
  flag=True
  for layer,act in zip(config[0],config[1]):  
      model.add(Dense(layer, activation=act))
      model.add(Dropout(0.1))

  model.add(Dense(l, activation='sigmoid'))

  opt = tensorflow.keras.optimizers.Adam(lr=0.0001, decay=1e-3)
  # Let's train the model using RMSprop
  model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy','categorical_accuracy','binary_accuracy'])
  callbacks = []
  #callbacks.append( EarlyStopping(monitor='val_loss',     min_delta=0, patience=10,  verbose=1, restore_best_weights=True) )
  #callbacks.append( ReduceLROnPlateau(monitor='val_loss', factor=1/3,  patience=3, verbose=1, min_delta=0.0001) )


  model.fit(X_train, y_train,
            epochs=n_epochs,
            batch_size=2048,callbacks=callbacks)
  y_pred = model.predict(X_val)
  #y_pred=oneHot(y_pred,classes)
  #scoreTrain=model.evaluate(X_train, y_train, batch_size=128)
  #scoreVal = model.evaluate(X_val, y_val, batch_size=128)
  #print(scoreTrain,scoreVal)
  #y_pred= 1*(y_pred>=np.mean(y_pred,axis=1).reshape(-1,1))
  y_pred= 1*(y_pred-np.mean(y_pred,axis=1).reshape(-1,1))/(np.std(y_pred,axis=1).reshape(-1,1))
  y_pred=1*(y_pred>=0)
  y_val=np.ravel(y_val)
  y_pred=np.ravel(y_pred)
  #print(y_val,y_pred)
  print('Validation Accuracy:',np.mean(y_val==y_pred))
  print('hamming_loss:',hamming_loss(y_val,y_pred))
  return model
#task7(x_train,y_train,x_test,y_test)

# In[ ]:




