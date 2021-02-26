import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, getsize, dirname
from collections import Counter, OrderedDict
import shutil
import warnings
import hickle
import h5py
import pickle
import gc
import random
import time
import feather
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
path = "../../../../../../zion/OpenSNP/people"
meta = "../../../../../../zion/OpenSNP/meta"
beacons = "../../../../../zion/OpenSNP/beacon"
main_path = join(beacons, "Main2")


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
    t = np.where(np.sum(beacon.values != "NN", axis=1) > 1)[0]
    beacon = beacon.iloc[t]
    ind = []
    for j in range(len(beacon.index)):
        if beacon.index[j][0] == "r":
            ind.append(j)
    ind = np.array(ind)
    beacon = beacon.iloc[ind]
    beacon.columns = beacon.columns.astype(int)
    return beacon, maf


# In[ ]:


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

# Read phenotype file
with open(join(beacons, "OpenSNP_Phenotype.pickle"), 'rb') as handle:
    pheno = pickle.load(handle)
print(pheno.shape)

# Trim people have less phenotypes than threshold
people_thres = 0
x = np.sum(pheno != "-", axis=1)
pheno = pheno.loc[x >= people_thres]
pheno.shape

files = [v for (k, v) in user_filename.items() if k in pheno.index.values]
print(len(files))

# 3033-100-200-300-400 --> 432 -- DONE
# 500-600-700-800-900 --> 900
# 1000-1100-1200-1300-1400 --> 1400
# 1500-1600-1700-1800-1900 --> 1900
# 2000-2100-2200-2300-2400 --> 2400
# 2500-2600-2700-2800-2900 --> 2900
# 3000 boşta kalıyor


j = 0
block_size = 5
ind = np.arange(100, 3001, 100)
for i in ind:
    if j % block_size == 0:
        j += 1
        with open(join(main_path, "tBeacon_main_" + str(i) + ".pickle"), 'rb') as handle:
            beacon = pickle.load(handle)
        gc.collect()
        print(" NEW START ", i, " is started --> ", beacon.shape)
        continue

    start = time.time()
    gc.collect()
    print("", i, " is started --> ", beacon.shape)
    with open(join(main_path, "tBeacon_main_" + str(i) + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    gc.collect()
    cp1 = time.time()
    print("Data is loaded in ", cp1 - start, " seconds")

    beacon = pd.merge(beacon, data, left_index=True, right_index=True, how="outer")
    gc.collect()
    cp2 = time.time()
    print("Merge is done in ", cp2 - cp1, " seconds")
    j += 1
    if j % block_size == 0:
        print("SAVING MERGINGS ", i)
        beacon.values[beacon.isnull().values] = "NN"
        gc.collect()
        cp3 = time.time()
        print("Filling NN is done in ", cp3 - cp2, " seconds")
        #beacon.to_pickle(join(beacons, "kBeacon_" + str(i) + ".pickle"))
        beacon.columns = beacon.columns.astype(str)
        gc.collect()
        beacon.reset_index(inplace=True)
        gc.collect()
        #beacon.to_parquet(join(beacons, "kBeacon_" + str(i) + ".parquet"), engine='fastparquet')
        beacon.to_pickle(join(beacons, "fBeacon_" + str(i) + ".pickle"))
        gc.collect()
        print()
