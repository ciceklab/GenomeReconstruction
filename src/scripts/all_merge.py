#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, getsize, dirname
from collections import Counter, OrderedDict
import shutil
import warnings
import h5py
import pickle
import gc
import random
import time
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
path = "../../../../../../zion/OpenSNP/people"
meta = "../../../../../../zion/OpenSNP/meta"
beacons = "../../../../../zion/OpenSNP/beacon"
main_path = join(beacons, "Main2")


# In[2]:


def findUserIndex(fileName):
    fileName = fileName[4:]
    return int(fileName.split("_")[0])


def findFileIndex(fileName):
    return int(fileName.split("_")[1][4:])


def findSkipCount(fileName):
    filePath = join(path, fileName)
    with open(filePath, "r") as f:
        i = 0
        for line in f:
            if line[0] == "#" or line[0] == " ":
                i += 1
            else:
                if line[0:6] == "rs-id":
                    i += 1
                break
        return i


def readClean(data):
    # Remove X,Y,MT chromosomes
    no_x_y = np.logical_and(data["chromosome"] != "X", data["chromosome"] != "Y")
    data = data[np.logical_and(no_x_y, data["chromosome"] != "MT")]
    data = data.fillna("NN")
    data[data == "II"] = "NN"
    data[data == "--"] = "NN"
    data[data == "DD"] = "NN"
    data[data == "DI"] = "NN"
    return data.iloc[np.where(data.iloc[:, [1]] != "NN")[0]]


def readDf(file, rowSkip):
    data = pd.read_csv(join(path, file), sep="\t", header=None, skiprows=rowSkip)
    data.columns = ['rs_id', 'chromosome', 'position', 'allele']
    del data['position']
    data = data.set_index('rs_id')
    data = data.rename(columns={"allele": findUserIndex(file)})
    return data


def readFileComplete(fileName):
    rowSkip = findSkipCount(fileName)
    beacon = readDf(fileName, rowSkip)
    beacon = readClean(beacon)
    return beacon


def mergeClean(beacon):
    beacon = beacon.loc[~beacon.index.duplicated(keep='first')]
    beacon = beacon[pd.to_numeric(beacon['chr'], errors='coerce').notnull()]
    beacon["chr"] = pd.to_numeric(beacon["chr"])
    beacon = beacon.sort_values(by=['chr'])
    beacon = beacon.fillna("NN")
    maf = beacon[['chr']]
    beacon = beacon.drop(['chr'], axis=1)
    ind = []
    for j in range(len(beacon.index)):
        if beacon.index[j][0] == "r":
            ind.append(j)
    ind = np.array(ind)
    beacon = beacon.iloc[ind]
    beacon.columns = beacon.columns.astype(int)
    return beacon, maf


# In[3]:


################################################################################################################################


# In[ ]:


# Trim except .23andme and small genome files
files = np.array([f for f in listdir(path) if isfile(join(path, f))], dtype=str)
types = []
sizes = []
for f in files:
    types.append(f.split(".")[-2])
    sizes.append(getsize(join(path, f)))
types = np.array(types)
sizes = np.array(sizes)
Counter(types)
ind = np.logical_and(types == "23andme", sizes > 15 * 1000000)
files = files[ind]

# Deal with multiple file people, select newest one
user_filename = {}
for f in files:
    user_filename.setdefault(int(findUserIndex(f)), []).append(f)
multiple_files = {k: v for (k, v) in user_filename.items() if len(v) > 1}

for m in multiple_files:
    f_names = multiple_files.get(m)
    selected = [findFileIndex(item) for item in f_names]
    selected = selected.index(max(selected))
    for i in range(len(f_names)):
        if i != selected:
            index = np.argwhere(files == f_names[i])
            files = np.delete(files, index)

user_filename = {}
for f in files:
    user_filename[int(findUserIndex(f))] = f
user_ind = np.array(list(user_filename.keys()))


# In[ ]:

'''
# Join Beacon's
with open(join(main_path, "tBeacon_main_" + str(3033) + ".pickle"), 'rb') as handle:
    beacon = pickle.load(handle)
gc.collect()

ind = np.arange(100, 3001, 100)

for i in ind:
    with open(join(main_path, "tBeacon_main_" + str(i) + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    gc.collect()
    #beacon = pd.merge(beacon, data, left_index=True, right_index=True, how="outer")
    beacon = pd.merge(beacon, data, left_index=True, right_index=True, how="outer")
    gc.collect()
    beacon.values[beacon.isnull().values] = "NN"
    gc.collect()
    print(i, " is completed.")

beacon.columns = beacon.columns.astype(str)
# Save to a file
beacon.to_parquet(join(beacons, "Beacon_3033.parquet"), engine='fastparquet')
gc.collect()    
#beacon.to_pickle(join(beacons, "Beacon_3033.pickle"))
print("Beacon is written")
beacon = None
data = None
'''

# Join MAF's
with open(join(main_path, "tMAF_main_" + str(3033) + ".pickle"), 'rb') as handle:
    maf = pickle.load(handle)
ind = np.arange(100, 3001, 100)
for i in ind:
    with open(join(main_path, "tMAF_main_" + str(i) + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    gc.collect()
    data = data.rename(columns={"chr": "chromosome"})
    maf = pd.merge(maf, data, left_index=True, right_index=True, how='outer')
    gc.collect()
    maf["chr"] = maf['chr'].fillna(maf['chromosome'])
    gc.collect()
    del maf["chromosome"]
    print(i, " is completed.")

maf.columns = maf.columns.astype(str)
maf = maf.astype(str)
gc.collect()
maf.to_pickle(join(beacons, "MAF_3034.pickle"))
#maf.to_parquet(join(beacons, "MAF_3033.parquet"), engine='fastparquet')

print("MAF is written")
maf = None
data = None
