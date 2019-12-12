#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
# create new folders inside train_dir
train_dir = 'C:/Users/Holt/Documents/MRMFinal/Faces'
nv = os.path.join(train_dir, 'Male')
os.mkdir(nv)
mel = os.path.join(train_dir, 'Female')
os.mkdir(mel)


# In[40]:


data_dir = 'C:/Users/Holt/Documents/MRMFinal/lfw-deepfunneled'

import pandas as pd

names = pd.read_csv('C:/Users/Holt/Documents/MRMFinal/female_names.txt', header = None, names = ["name"])


# In[41]:


names


# In[42]:


import os
import numpy as np
import shutil

def find_dir(name, path):
    for root, dirs, files in os.walk(path):
        if name in dirs:
            return os.path.join(root, name)
        
def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

names = np.array(names)
print(len(names))

for i in range(len(names)):
    file_name = names[i][0][:names[i][0].index('_0')]
    directory = find_dir(file_name, data_dir)
    if directory:
        dir_name = os.path.join(data_dir, names[i][0][:names[i][0].index('_0')])
        print(dir_name)
        files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
        for f in files:
            shutil.copy(os.path.join(directory, f), os.path.join('C:/Users/Holt/Documents/MRMFinal/Faces/Female', f))


# In[ ]:




