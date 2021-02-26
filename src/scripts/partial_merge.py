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
import random
import feather
import time
import gc
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


# 3033-100-200-300-400 --> 432+3000 == 3400 -- DONE
# 500-600-700-800-900 --> 900
# These are part 1

# 1000-1100-1200-1300-1400 --> 1400
# 1500-1600-1700-1800-1900 --> 1900
# These are part 2

# 2000-2100-2200-2300-2400 --> 2400
# 2500-2600-2700-2800-2900 --> 2900
# These are part 3


start = time.time() 
#data1 = pd.read_parquet(join(beacons, "kBeacon_900.parquet"), engine='fastparquet')
data1 = feather.read_dataframe(join(beacons, "fBeacon_500.ftr"))
data1.set_index("rs_id", inplace=True)
t = np.where(np.sum(data1.values != "NN", axis=1) > 1)[0]
gc.collect()
data1 = data1.iloc[t]

gc.collect()

cp1 = time.time()
print("Data 1 is loaded in ", cp1 - start, " seconds")

#data2 = pd.read_parquet(join(beacons, "kBeacon_3400.parquet"), engine='fastparquet')
data2 = feather.read_dataframe(join(beacons, "fBeacon_1000.ftr"))
data2.set_index("rs_id", inplace=True)
t = np.where(np.sum(data2.values != "NN", axis=1) > 1)[0]
gc.collect()
data2 = data2.iloc[t]
gc.collect()

cp2 = time.time()
print("Data 2 is loaded in ", cp2 - cp1, " seconds")


data1 = pd.merge(data1, data2, left_index=True, right_index=True, how="outer")
data2 = None
t = None
gc.collect()

cp3 = time.time()
print("Datas are merged in ", cp3 - cp2, " seconds")

#data1.values[data1.isnull().values] = "NN"
data1.fillna("NN", inplace=True)
gc.collect()

cp2 = time.time()
print("NN filled in ", cp2 - cp3, " seconds")

data1.columns = data1.columns.astype(str)
gc.collect()
#data1.to_parquet(join(beacons, "kBeacon_p1.parquet"), engine='fastparquet')
data1.reset_index(inplace=True)
data1.to_feather(join(beacons, "fBeacon_ps1.ftr"))
gc.collect()

cp3 = time.time()
print("Parqueted in ", cp3 - cp2, " seconds")



