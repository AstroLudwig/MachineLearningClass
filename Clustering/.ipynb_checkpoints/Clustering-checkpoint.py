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
num = 10
nnum = 5

X, y = make_blobs(n_samples=n_samples, random_state=random_state)

def get_inertia_predict(X,num):
	KM = KMeans(n_clusters=num, random_state=random_state,init="random") 
	y_pred = KM.fit_predict(X)
	inertia = KM.inertia_ 
	return inertia, y_pred

def get_ypred(X,num):
	KM = KMeans(n_clusters=num, random_state=random_state,init="random") 
	y_pred = KM.fit_predict(X)
	inertia = KM.inertia_ 

	return y_pred	

def get_inertia_range(X, num):
	inertia = np.zeros(num +1)
	for i in range(1,num+1):
		print(i)
		KM = KMeans(n_clusters=i, n_init=1, random_state=random_state,init="random").fit(X) 
		inertia[i] = KM.inertia_ 
	return inertia[1:]

def get_init(X, n_cluster,num,init):
	inertia = np.zeros(num +1)
	for i in range(1,num+1):
		print(i)
		KM = KMeans(n_clusters=n_cluster, n_init=i, random_state=random_state,init=init).fit(X) 
		inertia[i] = KM.inertia_ 
	return inertia[1:]


real_inertia = get_inertia_range(realX,num)
sim_inertia = get_inertia_range(X,num)

real_init_random = get_init(realX,2,nnum,"random")
sim_init_random = get_init(X,3,nnum,"random")
real_init_kmeans = get_init(realX,2,nnum,"k-means++")
sim_init_kmeans = get_init(X,2,nnum,"k-means++")

f, (ax,bx) = plt.subplots(1,2)
ax.scatter(realX[:,0],realX[:,1],c=get_ypred(realX,2))
ax.set_xlabel("g-r",size="large")
ax.set_ylabel("r-i",size="large")
bx.scatter(X[:, 0], X[:, 1],c=get_ypred(X,3))
bx.set_xlabel("Parameter 1",size="large")
bx.set_ylabel("Parameter 2",size="large")
ax.set_title("Quasar Data",size="large")
bx.set_title("Simulated Data",size="large")
plt.savefig("ExampleData.png")


f, (ax,bx) = plt.subplots(1,2)
ax.plot(real_inertia,range(1,num+1))
ax.set_ylabel("Number of Clusters",size="large")
ax.set_xlabel("Inertia",size="large")
ax.set_title("Quasar Data",size="large")

bx.plot(sim_inertia,range(1,num+1))
bx.set_xlabel("Inertia",size="large")
bx.set_title("Simulated Data",size="large")
plt.savefig("NumberOfClusters.png")

f, axes = plt.subplots(2,2)
# n-init, vs inertia
axes[0,0].plot(real_init_random,range(1,nnum+1))
axes[0,1].plot(sim_init_random,range(1,nnum+1))

axes[1,0].set_xlabel("inertia")
axes[1,1].set_xlabel("inertia")

axes[0,0].set_ylabel("n-init: random")
axes[1,0].set_ylabel("n-init: k-means++")

axes[0,0].set_title("Quasar Data",size="large")
axes[0,1].set_title("Simulated Data",size="large")

axes[1,0].plot(real_init_kmeans,range(1,nnum+1))
axes[1,1].plot(sim_init_kmeans,range(1,nnum+1))
plt.savefig("n-init.png")

plt.show()

