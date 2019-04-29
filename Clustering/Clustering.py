import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import random 
from random import randint 
from sklearn.datasets import make_blobs
from astroML.datasets import fetch_dr7_quasar
plt.style.use("seaborn")

# Real Data
data = fetch_dr7_quasar()
u_gr = data['mag_g'] - data['mag_r']
u_ri = data['mag_r'] - data['mag_i']
index = np.where(u_gr > -15)
u_gr = u_gr[index]
u_ri = u_ri[index]

realX = np.zeros([len(u_gr),2])
realX[:,0] = u_gr
realX[:,1] = u_ri

# Fake Data
n_samples = 1500
random_state = 170

X, y = make_blobs(n_samples=n_samples, random_state=random_state)

def get_inertia(X,num):
	KM = KMeans(n_clusters=num, random_state=random_state) 
	y_pred = KM.fit_predict(X)
	inertia = KM.inertia_ 

	return inertia

def get_ypred(X,num):
	KM = KMeans(n_clusters=num, random_state=random_state) 
	y_pred = KM.fit_predict(X)
	inertia = KM.inertia_ 

	return y_pred

inertia = [get_inertia(X,i) for i in range(1,10)]


f, (ax,bx) = plt.subplots(1,2)
ax.scatter(realX[:,0],realX[:,1],c=get_ypred(realX,2))
bx.scatter(X[:, 0], X[:, 1],c=get_ypred(X,3))


plt.figure()
plt.plot(range(1,10),inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Simulated Data")
plt.show()