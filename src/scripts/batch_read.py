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
import random
import time
import gc
import feather
warnings.filterwarnings('ignore')
path = "../../../../../zion/OpenSNP/people"
meta = "../../../../../zion/OpenSNP/meta"
beacons = "../../../../zion/OpenSNP/beacon"


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
    no_x_y = np.logical_and(data["chromosome"] != "X",data["chromosome"] != "Y")
    data = data[np.logical_and(no_x_y, data["chromosome"] != "MT")]
    data = data.fillna("NN")
    data[data == "II"] = "NN"
    data[data == "--"] = "NN"
    data[data == "DD"] = "NN"
    data[data == "DI"] = "NN"
    return data.iloc[np.where(data.iloc[:,[1]] != "NN")[0]]

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
    t = np.where(np.sum(beacon.values != "NN", axis=1) > 0)[0]
    beacon = beacon.iloc[t]
    ind = []
    for j in range(len(beacon.index)):
        if beacon.index[j][0] == "r":
            ind.append(j)
    ind = np.array(ind)
    beacon = beacon.iloc[ind]
    beacon.columns = beacon.columns.astype(str)
    return beacon, maf

# Trim except .23andme and small genome files
files = np.array([f for f in listdir(path) if isfile(join(path, f))], dtype=str)
types = []
sizes = []
for f in files:
    types.append(f.split(".")[-2])
    sizes.append(getsize(join(path,f)))
types = np.array(types)
sizes = np.array(sizes)
Counter(types)
ind = np.logical_and(types == "23andme", sizes > 15 * 1000000)
files = files[ind]

# Deal with multiple file people, select newest one
user_filename = {}
for f in files:
    user_filename.setdefault(int(findUserIndex(f)),[]).append(f)
multiple_files = {k:v for (k,v) in user_filename.items() if len(v) > 1}

for m in multiple_files:
    f_names = multiple_files.get(m)
    selected = [findFileIndex(item) for item in f_names]
    selected = selected.index(max(selected))
    for i in range(len(f_names)):
        if i != selected:
            index = np.argwhere(files==f_names[i])
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


files = [v for (k,v) in user_filename.items() if k in pheno.index.values]
len(files)

features = ['ColorBlindness',
            'HairType',
            'HairColor',
            'EyeColor',
            'TanAbility',
            'Asthma',
            'LactoseIntolerance',
            'BloodType',
            'EarWax',
            'Freckling',
            'TongueRoller',
            'RingFinger',
            'BeardColor',
            'Intolerance']

# BUILD BEACON
main_path = join(beacons, "Main")
print("Started main beacon build.")
beacon = readFileComplete(files[0])
beacon = beacon.rename(columns={"chromosome": "chr"})
gc.collect()
i = 1
while i < len(files):
    start = time.time()
    try:
        data = readFileComplete(files[i])
        gc.collect()
        beacon = pd.merge(beacon, data, left_index=True, right_index=True, how='outer')
        gc.collect()
        beacon["chr"].fillna(beacon["chromosome"], inplace=True)
        gc.collect()
        beacon = beacon.drop("chromosome", axis=1)
    except:
        print("File " + files[i] + " is skipped.\n")
    end = time.time()
    print(str(i) + ". step is completed in " + str(end - start) + " seconds.")

    if i % 100 == 0 or i == len(files) - 1:
        print("Cleaning main beacon started.")
        beacon, maf = mergeClean(beacon)
        print("Cleaned main beacon.")
        # SAVE
        gc.collect()
        #beacon.reset_index(inplace = True) 
        #maf.reset_index(inplace = True) 
        beacon.to_pickle(join(main_path, "sBeacon_main_"+str(i)+".pickle"))
        maf.to_pickle(join(main_path, "sMAF_main_"+str(i)+".pickle"))
        gc.collect()
        if i != len(files) - 1:
            i+=1
            print("\n" + str(i) + " has started")
            beacon = readFileComplete(files[i])
            gc.collect()
            beacon = beacon.rename(columns={"chromosome": "chr"})
    i+=1
print("Ended main beacon build.\n")

print([item for item, count in Counter(beacon.index.values).items() if count > 1])