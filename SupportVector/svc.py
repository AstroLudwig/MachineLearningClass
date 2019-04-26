import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import random 
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D 

def get_tests(clf_func,test_x,test_y,label):
	
	# Try out your predictions 
	predict_y = clf_func.predict(test_x)

	# Were your predictions true or false? 
	true_labels = predict_y[predict_y == test_y]
	false_labels = predict_y[predict_y != test_y]

	"I think it's a blue star and it is a blue star."
	TruePositive = len(true_labels[true_labels == label])

	"I don't think it's a blue star, but it is a blue star."
	FalseNegative = len(false_labels[false_labels != label])

	"I think it's a blue star, but it's not a blue star."
	FalsePositive = len(false_labels[false_labels == label])

	Completeness = TruePositive / (TruePositive + FalseNegative)
	Purity = TruePositive / (TruePositive + FalsePositive)

	return Completeness, Purity 

contamination = 3
# Generate Random Test Data

noise = np.random.normal(0,1,100)

# Team 1, Blue Stars 

blue_y = np.arange(10, 15, .05) 
blue_x = np.arange(.1, 3, (3-.1)/len(blue_y)) 
blue_z = np.arange(4,9,(9-4)/len(blue_y))

random.shuffle(blue_y); random.shuffle(blue_x)
random.shuffle(blue_z)

# Team 2, Black Diamonds

black_y = np.arange(5, 12, .05) 
black_x = np.arange(2.7, 6, (6-2.7)/len(black_y)) 
black_z = np.arange(4.5,11,(11-4.5)/len(black_y))

random.shuffle(black_y); random.shuffle(black_x)
random.shuffle(black_z)

data = np.zeros([len(blue_y)+len(black_y),2])
data[:,0] = np.append(blue_x,black_x)
data[:,1] = np.append(blue_y,black_y)
labels = np.append(np.repeat("Blue",len(blue_y)),np.repeat("Black",len(black_y)))


kf = KFold(n_splits=2,shuffle=True)
for train_index, test_index in kf.split(data[:,0]):
	train_x, test_x = data[train_index], data[test_index]
	train_y, test_y = labels[train_index], labels[test_index]

param_x,param_y = np.meshgrid(np.arange(0,7,.01),np.arange(3,16,.01))

# Linear Fit
svc = SVC(C= contamination, kernel='linear')
svc.fit(train_x,train_y)

decision_function_lin = svc.decision_function(np.c_[param_x.ravel(),param_y.ravel()])


comp_blue, pure_blue = get_tests(svc,data,labels,"Blue")
comp_black, pure_black = get_tests(svc,data,labels,"Black")

# Poly Fit

svc_curv = SVC(C= contamination, kernel='poly')

svc_curv.fit(train_x,train_y)

decision_function_curv = svc_curv.decision_function(np.c_[param_x.ravel(),param_y.ravel()])


comp_blue_curv, pure_blue_curv = get_tests(svc_curv,data,labels,"Blue")
comp_black_curv, pure_black_curv = get_tests(svc_curv,data,labels,"Black")


# Plot the data set with decision boundary, margins, and label the support vectors from SVM.
f, (ax,bx) = plt.subplots(1,2)
# Linear 

ax.contour(param_x,param_y,decision_function_lin.reshape(param_x.shape),[-1,0,1])
ax.scatter(blue_x,blue_y,marker="*",c="blue")
ax.scatter(black_x,black_y,marker="D",c="black")
ax.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],edgecolors="red",marker="o",facecolors='none',s=120)
ax.set_xlim(.5,5.5)
ax.set_ylim(4,14)
ax.set_title(("Contamination: {}, Kernel: linear \n Blue Completess: {:.3f} Purity: {:.3f} \n Black Completess: {:.3f} Purity: {:.3f}").format(contamination,comp_blue, pure_blue,comp_black, pure_black))

# Poly

bx.contour(param_x,param_y,decision_function_curv.reshape(param_x.shape),[-1,0,1])
bx.scatter(blue_x,blue_y,marker="*",c="blue")
bx.scatter(black_x,black_y,marker="D",c="black")
bx.scatter(svc_curv.support_vectors_[:,0],svc_curv.support_vectors_[:,1],edgecolors="red",marker="o",facecolors='none',s=120)
bx.set_xlim(.5,5.5)
bx.set_ylim(4,14)
bx.set_title(("Contamination: {}, Kernel: poly \n Blue Completess: {:.3f} Purity: {:.3f} \n Black Completess: {:.3f} Purity: {:.3f}").format(contamination,comp_blue_curv, pure_blue_curv,comp_black_curv, pure_black_curv))

f.suptitle("Levels: -1,0,1")


plt.show()
