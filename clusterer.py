#!/usr/bin/python3

import sys
sys.path.append("/home/constantin/Documents/practica/")

from sklearn.cluster import AffinityPropagation
import csv
import numpy as np

trainSetFile = './matrix.csv'

##############################################################################

X=[]
with open(trainSetFile, 'rU') as f:
    reader = csv.reader(f,delimiter=',')
    matrix = [rec for rec in reader]
f.close()

data = [rec[1:] for rec in matrix]
X = [rec[0] for rec in matrix]

X = np.asarray(X)

##############################################################################
# Compute Affinity Propagation


af = AffinityPropagation(preference=-59999)
#af.affinity_matrix_ = data
#af.affinity = "precomputed"

af.fit_predict(data)
labels = af.labels_
cluster_centers_indices = af.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
##############################################################################
# View Results
n = n_clusters_

centers = []

for k in range (n_clusters_):
    print()
    print(' --  CLUSTER '  + str(k+1) + '  --  ')
    print()
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]

    if len(X[class_members]) < 10:
        n -= 1

    for drug in X[class_members]:
        if drug == cluster_center:
            centers.append(drug)
            print(' - ' + drug.upper() + ' - ')
        else:
            print(drug)