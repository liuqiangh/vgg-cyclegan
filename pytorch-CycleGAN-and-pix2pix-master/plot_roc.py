import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve,auc ###计算roc

A_pre_testA = np.load("A_pre_testA.npy")
A_pre_testB = np.load("A_pre_testB.npy")
B_pre_testA = np.load("B_pre_testA.npy")
B_pre_testB = np.load("B_pre_testB.npy")
pre_testA = B_pre_testA
pre_testB = B_pre_testB
#pre_testA = A_pre_testA - B_pre_testA
#pre_testB = A_pre_testB - B_pre_testB
#pre_testA = np.load("pre_testA.npy")
#pre_testB = np.load("pre_testB.npy")

pre_test = np.append(pre_testA,pre_testB)

len_pre_testA = len(pre_testA)
len_pre_testB = len(pre_testB)
len_pre_test = len(pre_test)
label_test = np.zeros((len_pre_test))
label_test[1:len_pre_testA] = 0
label_test[len_pre_testA+1:len_pre_test] = 1
fpr, tpr, thresholds = metrics.roc_curve(label_test, pre_test,pos_label=1)
roc_auc = auc(fpr, tpr)

print(roc_auc)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
#plt.show()
plt.savefig('D_B_roc.png')