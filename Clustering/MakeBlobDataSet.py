import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import random 
from random import randint 
# """""""""""""""""""""""""""""""""""""""""
# Create Fake Data 
# """""""""""""""""""""""""""""""""""""""""

def make_clump(P_x,P_y,x0,y0,max_d):

	n = 30

	num_points = 70 

	init = max_d / n

	x,y = np.meshgrid(np.arange(P_x[0],P_x[1],.1),np.arange(P_y[0],P_y[1],.1))
	x = x.flatten(); y = y.flatten()

	d = np.sqrt((x-x0)**2+(y-y0)**2)

	index = np.where(d < max_d)

	x = x[index]; y =y[index]; d = d[index]
	
	random_index = [randint(0,int(len(x))-1) for p in range(len(d)-1)]
	x = x[random_index]; y =y[random_index]; d = d[random_index]

	points = []; counter = 0

	for i in range(len(d)):
		if (d[i] < init) and (counter < .99 * len(d)):
			points.append([x[i],y[i]])
		if (d[i] < 2 * init) and (d[i] > init) and (counter < .9 * len(d)):
		 	points.append([x[i],y[i]])		
		if (d[i] < 3 * init) and (d[i] > 2 * init) and (counter < .8 * len(d)):
		 	points.append([x[i],y[i]])
		if (d[i] < 4 * init) and (d[i] > 3 * init) and (counter < .75 * len(d)):
		 	points.append([x[i],y[i]])	
		if (d[i] < 5 * init) and (d[i] > 4 * init) and (counter < .7 * len(d)):
		 	points.append([x[i],y[i]])	 	 			

		counter += 1
	points_x = np.array(points)[:,0]
	points_y = np.array(points)[:,1]


	random_index = [randint(0,int(len(points_x))) for p in range(num_points-1)]
	points_x = points_x[random_index]; points_y =points_y[random_index]

	return points_x,points_y

test = False
if test:
	param_xrange = [-100,100]
	param_yrange = [-100,100]

	centers = [[20,43],[30,70],[40,30]]
	dict = {}
	dict["Cx_1"], dict["Cy_1"] = make_clump(param_xrange,param_yrange,20,43,50)
	dict["Cx_2"], dict["Cy_2"] = make_clump(param_xrange,param_yrange,33,45,50)
	dict["Cx_3"], dict["Cy_3"] = make_clump(param_xrange,param_yrange,30,33,50)


	plt.style.use("seaborn")
	plt.figure()
	plt.scatter(dict["Cx_1"],dict["Cy_1"],s=10,c="black")
	plt.scatter(dict["Cx_2"],dict["Cy_2"],s=10,c="blue")	
	plt.scatter(dict["Cx_3"],dict["Cy_3"],s=10,c="red")						
	plt.xlim(0,50 ); 
	plt.ylim(20,60)
	plt.xlabel("Parameter 1"); plt.ylabel("Parameter 2")
	plt.show()