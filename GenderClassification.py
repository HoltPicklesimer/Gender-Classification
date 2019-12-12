#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


DATADIR = "C:/Users/Holt/Documents/MRMFinal/Faces"
CATEGORIES = ["male", "female"]
IMG_SIZE = 50


# In[3]:


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to dir
        class_num = CATEGORIES.index(category)
        male_indices = random.sample(range(1, 10272), 2966)
        i = 0
        for img in os.listdir(path): # male_indices and i used to get random male images since there are more males than females
            i += 1
            if category == "female" or i in male_indices:
                try:
                    img_array = cv2.imread(os.path.join(path,img))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    print(e)


create_training_data()


# In[5]:


random.shuffle(training_data)


# In[7]:


X = []
y = []


# In[8]:


for features, label in training_data:
    X.append(features)
    y.append(label)

np.shape(X)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# In[9]:


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[10]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


# In[12]:


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)


# In[15]:


img_array = cv2.imread('C:/Users/Holt/Documents/MRMFinal/Ethan.PNG')
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
X = new_array
y = 0
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = X/255.0
prediction = model.predict(X)
print(prediction)


# In[16]:


# Do it with no color, just black and white
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to dir
        class_num = CATEGORIES.index(category)
        male_indices = random.sample(range(1, 10272), 2966)
        i = 0
        for img in os.listdir(path): # male_indices and i used to get random male images since there are more males than females
            i += 1
            if category == "female" or i in male_indices:
                try:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    print(e)


create_training_data()


# In[17]:


random.shuffle(training_data)


# In[19]:


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

np.shape(X)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[20]:


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[21]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


# In[32]:


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)


# In[31]:


img_array = cv2.imread('C:/Users/Holt/Documents/MRMFinal/Rochak.PNG', cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
X = new_array
y = 0
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X/255.0
prediction = model.predict(X)
if prediction > 0.5:
    print("Female")
else:
    print("Male")


# In[ ]:




