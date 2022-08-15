import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
from sklearn import datasets
import numpy as np
import h5py
from sklearn.cluster import KMeans
from torch.utils.data import Dataset


path = 'F:\\Python code\\venv\\share\man\SDCN\predict.txt'
pred = np.loadtxt(path,dtype=float)
# pred = np.zeros((195,500))
# k = -1
# j = 0
# for i in range(0,len(x)):
#     if(i%500==0):
#         k += 1
#         j = 0
#     pred[k][j] = x[i]
#     j += 1

path = 'F:\\Python code\\venv\\share\man\SDCN\\res\clures.txt'
y = np.loadtxt(path,dtype=int)
path = 'F:\\Python code\\venv\\share\man\SDCN\\data\\hebing.txt'
x = np.loadtxt(path,dtype=float)

kmeans_model = KMeans(n_clusters=3, random_state=2).fit(pred)
labels = kmeans_model.labels_
print('predsil:',metrics.silhouette_score(pred, y)) #越接近1越好
print('predcal:',calinski_harabasz_score(pred,y))  #越大越好
print('preddb',metrics.davies_bouldin_score(pred, y)) #越小越好
# print('predari:',metrics.adjusted_mutual_info_score(pred,y)) #ari
# print('predvmeasure:',metrics.v_measure_score(pred,y))
print('xsil:',metrics.silhouette_score(x, y)) #越接近1越好
print('xcal:',calinski_harabasz_score(x,y))  #越大越好
print('xdb',metrics.davies_bouldin_score(x, y)) #越小越好
# print('xari:',metrics.completeness_score(x,y))
# print('xvmeasure:',metrics.v_measure_score(x,y))