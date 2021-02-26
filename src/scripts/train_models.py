#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import random
import math
import timeit
import itertools
import warnings
import pickle
import feather
import gc
import sys
import os
import matplotlib.pyplot as plt
from os.path import join, isfile
from collections import Counter
from xgboost import XGBClassifier
from fcmeans import FCM
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, classification_report, mutual_info_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, formatter={'float': lambda x: "{0:0.2f}".format(x)})

# In[ ]:

mainPath = "../../../data"
beacons = join(mainPath, "beacon")
testSets = join(beacons, "testsets")
models = join(mainPath, "models")
ceuPath = join(beacons, "CEU")
opensnpPath = join(beacons, "OpenSNP")


# #### STEP 1: Load Beacon, MAF, Reference and other cached variables

# In[ ]:

features = ['EyeColor', 'HairType', 'HairColor', 'TanAbility', 'Asthma', 'LactoseIntolerance',  # 'BloodType',
            'EarWax', 'Freckling', 'TongueRoller', 'RingFinger', 'Intolerance', 'WidowPeak', 'ADHD', 'Acrophobia',
            'FingerHair', 'Myopia', 'IrritableBowel', 'IndexLongerBig', 'Photoptarmis', 'Migraine', 'RhProtein']
with open(join(opensnpPath, "OpenSNP_Phenotype.pickle"), 'rb') as handle:
    pheno = pickle.load(handle)
pheno = pheno[features]
pheno[pheno == "Auburn"] = "Blonde"
pheno[pheno == "Black"] = "Brown"

with open(join(opensnpPath, "MAF.pickle"), 'rb') as handle:
    maf = pickle.load(handle)

with open(join(opensnpPath, "Reference.pickle"), 'rb') as handle:
    reference = pickle.load(handle)
reference = reference.values

with open(join(opensnpPath, "Beacon.pickle"), 'rb') as handle:
    beacon = pickle.load(handle)

with open(join(opensnpPath, "BinaryBeacon.pickle"), 'rb') as handle:
    binary = pickle.load(handle)

with open(join(opensnpPath, "TernaryBeacon.pickle"), 'rb') as handle:
    ternary = pickle.load(handle)


# #### Constrainted Indices

# In[ ]:

pheno5People = pheno.iloc[np.where(np.sum(pheno != "-", axis=1) >= 10)[0]].index
pheno5People = pheno5People.map(str)
pheno5People = np.where(beacon.columns.isin(pheno5People))[0]

pheno1People = pheno.iloc[np.where(np.sum(pheno != "-", axis=1) >= 1)[0]].index
pheno1People = pheno1People.map(str)
pheno1People = np.where(beacon.columns.isin(pheno1People))[0]

phenoAllPeople = np.arange(beacon.shape[1])


# #### STEP 1.2: Function Definitions

# In[ ]:

# Beacon operations
def queryBeacon(beacon_people):
    return binary[:, beacon_people].any(axis=1)


def getMutationAt(index):
    temp = maf.iloc[index]
    if temp["minor_freq"] == temp["maf"]:
        return temp["minor"] + temp["minor"]
    else:
        return temp["major"] + temp["major"]


def div(n, d):
    return n / d if d else 0


def rpaCalculate(tp, fp, tn, fn):
    recall = div(tp, (tp + fn))
    precision = div(tp, (tp + fp))
    accuracy = div((tp + tn), (tp + fp + tn + fn))
    return recall, precision, accuracy


def getTrainingData(phenotype, pos, test_people):
    # Find indices of people who has the specified feature
    feature_label = pheno[pheno[phenotype] != "-"][phenotype]
    existing = beacon.columns.isin(feature_label.index.values)
    existing[test_people] = False

    # Get training data
    X = binary[pos][:, existing].T
    Y = feature_label[beacon.columns[existing]].values
    return X, Y

# Performance method


def performance(person, reconstruction, eval_pos, reference):
    ind = np.logical_and(person[eval_pos] != np.squeeze(reference)[eval_pos], person[eval_pos] != "NN")
    tp = np.sum(reconstruction[eval_pos][ind] != np.squeeze(reference)[eval_pos][ind])
    fn = np.sum(ind) - tp
    fp = np.sum(reconstruction[eval_pos][~ind] != np.squeeze(reference)[eval_pos][~ind])
    tn = np.sum(~ind) - fp

    return tp, fp, tn, fn


def performance_f(test_people, reconstructed, add_count, cluster_count, eval_pos):
    total_values = np.zeros((4))
    best_matches = []
    # For all people in victim set
    for i in range(add_count):
        all_combinations = np.zeros((4, cluster_count))
        rpa = np.zeros((3, cluster_count))
        # For each cluster obtained
        for j in range(cluster_count):
            all_combinations[:, j] = performance(test_people[i], reconstructed[j], eval_pos, reference)
            rpa[:, j] = rpaCalculate(*all_combinations[:, j])
        ind = np.argmax(rpa[0, :] * rpa[1, :])  # Best-match index
        best_matches.append(ind)
        total_values += all_combinations[:, ind]  # Add total tp-fp-tn-fn
    recall, precision, accuracy = rpaCalculate(*total_values)
    print("Recall_Micro_Avg    =", round(recall, 2), "\nPrecision_Micro_Avg =", round(precision, 2))
    return (precision, recall, accuracy), total_values, best_matches


# #### STEP 2: Choose random people and send query to Beacon to obtain No-Yes answers

# In[ ]:

def getNoYes(add_count, beacon_size):
    added_people = pheno5People.copy()
    random.shuffle(added_people)
    added_people = added_people[:add_count]
    beacon_people = np.setdiff1d(phenoAllPeople, added_people)
    random.shuffle(beacon_people)
    beacon_people = beacon_people[:beacon_size]
    before = queryBeacon(beacon_people)
    updated_beacon = np.concatenate([added_people, beacon_people])
    after = queryBeacon(updated_beacon)
    no_yes_indices = np.where(np.logical_and(before == False, after == True))[0]
    yes_yes_indices = np.where(np.logical_and(before == True, after == True))[0]
    print("Number of No-Yes SNP's : ", len(no_yes_indices))

    return yes_yes_indices, no_yes_indices, added_people


# #### STEP 3: Correlation Model

# In[ ]:


def builtSNPNetwork(no_yes_indices, model_ind, reference):
    model = ternary[no_yes_ind][:, model_ind].astype(float)
    model[model == -1] = np.nan
    x = pairwise_distances(model, metric="sokalmichener", n_jobs=-1)
    x = 1 - np.nan_to_num(x)
    return x


# ##### Spectral Clustering

# In[ ]:


def spectralClustering(no_yes_indices, add_count, x, reference, cluster_count=None):
    if not cluster_count:
        cluster_count = add_count
    sc = SpectralClustering(cluster_count, affinity='precomputed', n_init=100, n_jobs=-1)
    sc.fit(np.array(x))
    bins = []
    for i in range(cluster_count):
        temp = []
        for element in np.where(sc.labels_ == i)[0]:
            temp.append(no_yes_indices[element])
        #print("Bin " + str(i) + " has " + str(len(temp)) + " SNP's")
        bins.append(temp)
    reconstructed = np.array([reference.T[0] for i in range(cluster_count)])
    for i in range(cluster_count):
        for j in bins[i]:
            reconstructed[i][j] = getMutationAt(j)
    return reconstructed

# ## Phenotype Prediction

# In[ ]:


def evaluate_ensemble(models, x_test, y_test, add_count, cluster_count):
    results = []
    for i in models:
        results.append(i[1].predict_proba(x_test))
    labels = [i[0] for i in models]

    top3, top1 = 0, 0
    for i in range(add_count):
        test_person = y_test[labels].iloc[i]
        available_phenotypes = np.where(test_person != "-")[0]
        probs = np.zeros((cluster_count))
        for j in range(cluster_count):
            for k in available_phenotypes:
                target_label_ind = np.where(models[k][1].classes_ == test_person[k])[0]
                probs[j] += results[k][j][target_label_ind] * models[k][2]
        matched_ind = np.argsort(probs)[-3:]
        print(probs, "\n", matched_ind, "--", matches[i], "\n")
        if matches[i] in matched_ind:
            top3 += 1
        if matches[i] == matched_ind[-1]:
            top1 += 1
    print("Top-1 Accuracy= ", top1 / add_count, "\tTop-3 Accuracy= ", top3 / add_count)
    return top1 / add_count, top3 / add_count


# In[ ]:

def train_models(train_snps, test_people):
    models = []
    count = 1
    for feature in features:
        X, Y = getTrainingData(phenotype=feature, pos=train_snps, test_people=test_people)
        print("\n", count, ".", feature, "\tlabels=", np.unique(Y))
        X, Y = SMOTE().fit_sample(X, Y)
        rf = RandomForestClassifier(class_weight='balanced_subsample', oob_score=True, n_jobs=-1)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=16)
        model = GridSearchCV(cv=cv, estimator=rf, scoring='f1_macro', param_grid=parameters, verbose=0, n_jobs=-1)
        result = model.fit(X, Y)

        print("Best: %f using %s" % (result.best_score_, result.best_params_))
        best_model = result.best_estimator_
        best_score = (result.best_score_ + best_model.oob_score_) / 2
        if best_score > 1.2 / len(np.unique(Y)):
            count += 1
            print("Train:", round(best_model.score(X, Y), 2), " | Validation:", round(best_score, 2))
            models.append((feature, model, best_score))
    return models


# In[ ]:


Estimators = [100]                 # n_estimators
Depths = [3, 4, 6, None]          # max_depth (None olabilir)
MinSample = [0.01, 0.04]           # min_samples_leaf
MaxFeatures = ["auto", 1 / 2, 1 / 4]  # min_samples_leaf
Criterion = ["gini"]              # criterion
parameters = {"max_depth": Depths, "min_samples_leaf": MinSample, "criterion": Criterion, "n_estimators": Estimators, "max_features": MaxFeatures}

# #### All

# In[ ]:


experiments = [(2, 20, 0.9), (3, 30, 0.8), (5, 50, 0.8), (10, 100, 0.8), (20, 100, 0.65)]
res = []
for e in experiments:
    add_count = e[0]
    beacon_size = e[1]
    with open(join(testSets, str(add_count) + "_testset.pkl"), 'rb') as f:
        test_sets = pickle.load(f)
    top1s = []
    top3s = []
    for i in range(10):
        yes_yes_ind, no_yes_ind, added_people = test_sets[i]
        model_ind = np.setdiff1d(pheno1People, added_people)

        # Genome Reconstruction
        correlations = builtSNPNetwork(no_yes_ind, model_ind, reference)
        reconstructed_spectral = spectralClustering(no_yes_ind, add_count, correlations, reference)
        (precision, recall, accuracy), _, matches = performance_f(beacon.iloc[:, added_people].values.T, reconstructed_spectral, add_count, add_count, no_yes_ind)

        # Phenotype Prediction
        models = train_models(train_snps=no_yes_ind, test_people=added_people)

        # Test Data
        x_test = (reconstructed_spectral[:, no_yes_ind] != reference[no_yes_ind].T).astype(np.int8)
        y_test = pheno.loc[beacon.columns[added_people]]

        # Performance
        top1, top3 = evaluate_ensemble(models, x_test, y_test, add_count, add_count)
        top1s.append(top1)
        top3s.append(top3)
    print("Top-1= ", np.mean(top1s), "\tTop-3= ", np.mean(top3s))
    res.append((top1s, top3s))
    with open(join(beacons, str(add_count) + ".pkl"), 'wb') as f:
        pickle.dump((top1s, top3s), f)


# In[ ]:


top1s = []
top3s = []
for i in [2, 5, 10, 20]:
    with open(join(beacons, str(i) + ".pkl"), 'rb') as f:
        t = pickle.load(f)
    top1s.append(np.mean(t[0]))
    top3s.append(np.mean(t[1]))
plt.ylabel("Accuracy")
plt.xlabel("Number of Added People")
plt.yticks(np.arange(0, 1.06, 0.05))
plt.ylim(0, 1.04)
my_xticks = ['2', '5', '10', '20']
plt.xticks(np.array([0, 1, 2, 3]), my_xticks)
plt.plot(top1s, label="Top-1")
plt.plot(top3s, label="Top-3")
plt.legend(loc="upper right")
plt.grid()
plt.savefig('phenotype.jpg', format='jpg', dpi=300)
