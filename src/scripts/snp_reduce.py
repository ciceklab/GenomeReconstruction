#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, getsize, dirname
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
import shutil
import warnings
import hickle
import h5py
import pickle
import random
import gc
import time
import feather
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
path = "../../../../../../zion/OpenSNP/people"
meta = "../../../../../../zion/OpenSNP/meta"
beacons = "../../../../../zion/OpenSNP/beacon"
main_path = join(beacons, "Main")


# In[3]:


'''
with open(join(beacons, "RMAF_3034.pickle"), 'rb') as handle:
    maf = pickle.load(handle)


# In[4]:


ind = np.arange(500, 3001, 500)

for i in ind:
    data = feather.read_dataframe(join(beacons, "fBeacon_" + str(i) + ".ftr"))
    data.set_index("rs_id", inplace=True)
    print("Read in: ", data.shape)
    gc.collect()
    data = maf.join(data, how="left")
    gc.collect()
    del data["chr"]
    del data["count"]
    gc.collect()
    data.fillna("NN", inplace=True)
    gc.collect()
    print("Writing in: ", data.shape)
    data.reset_index(inplace=True)
    data.to_feather(join(beacons, "RBeacon_" + str(i) + ".ftr"))

ind = np.arange(1000,3001,500)
beacon = feather.read_dataframe(join(beacons, "fBeacon_"+str(500)+".ftr"))
beacon.set_index("rs_id", inplace=True)
gc.collect()
'''


ind = np.arange(1000, 3001, 500)
beacon = feather.read_dataframe(join(beacons, "RBeacon_" + str(500) + ".ftr"))
beacon.set_index("rs_id", inplace=True)
gc.collect()

for i in ind:
    data = feather.read_dataframe(join(beacons, "RBeacon_" + str(i) + ".ftr"))
    data.set_index("rs_id", inplace=True)
    print("Read in: ", data.shape)
    gc.collect()

    beacon = pd.concat([beacon, data], axis=1)
    gc.collect()

print("Writing in: ", data.shape)
beacon.reset_index(inplace=True)
beacon.to_feather(join(beacons, "Beacon.ftr"))
