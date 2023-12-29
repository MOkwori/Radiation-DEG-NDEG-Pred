# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:31:47 2021

@author: Michael
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2,l1
from keras.optimizers import Adam,Adagrad,Adamax
import pickle
import collections
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

import tensorflow
import gc
gc.disable()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import itertools
import scipy.io as sio
import ckwrap


# load feature lists
feature_list =  pickle.load(open('feature_list.pkl', 'rb'))
featurs_s =  pickle.load(open('featurs_s.pkl', 'rb'))
featurs_e =  pickle.load(open('featurs_e.pkl', 'rb'))
featurs_t =  pickle.load(open('featurs_t.pkl', 'rb'))
featurs_m =  pickle.load(open('featurs_m.pkl', 'rb'))
featurs_c =  pickle.load(open('featurs_c.pkl', 'rb'))

#cleanup feature 
featurs_m.remove('mmu-miR-196a-2-3p')    # remove feature that was constant for all of new gene collection
featurs_m.append('miRNA')     # add missing miRNA feature

#load data of selected percentage of DE (30% and 100% plotted in Figure 5 of paper)
data_X =  pickle.load(open('data_X_30.pkl', 'rb'))
data_Y =  pickle.load(open('data_Y_30.pkl', 'rb'))

# data_X =  pickle.load(open('data_X_100.pkl', 'rb'))
# data_Y =  pickle.load(open('data_Y_100.pkl', 'rb'))

#generate subsets of dataset for Groups of features
dataset = pd.DataFrame(data_X, columns=feature_list)
data_X_s = dataset.loc[:,featurs_s]
data_X_e = dataset.loc[:,featurs_e]
data_X_t = dataset.loc[:,featurs_t]
data_X_m = dataset.loc[:,featurs_m]
data_X_c = dataset.loc[:,'chromatin_state']
data_X_r = dataset.loc[:,'is_RBP']
data_X_tt = dataset.loc[:,['Adrenal','Eye','Kidney','Liver','Muscle']]   #add tissue type feature set



auc_array=[] 
auprc_array = []
avg_precision = []
avg_recall= []
avg_f1 = []
avg_MCC= []




probas_XGB = []
Y_grand = []

scores=[]
scores2=[]
f_selection =0
no_epochs=100
# no_features =20
# if f_selection == 0:
#     no_features=len(data_X[0])
rg_f = 0.001
dr_f =0.04 
rr= 42 #randrange(100) #5#None #42   #21


comn_opt = ['data_X_s','data_X_e', 'data_X_c', 'data_X_t', 'data_X_m','data_X_r','data_X_tt']


dfs_in_list = [data_X_s,data_X_e, data_X_c, data_X_t, data_X_m, data_X_r,data_X_tt]

see=[]
combinations = []
combination_names =[]

for length in range(1, len(dfs_in_list)+1):
    combinations.extend(list(itertools.combinations(dfs_in_list, length)))
    combination_names.extend(list(itertools.combinations(comn_opt, length)))
    
count =0
for c in combinations:
    data_X=pd.concat(c, axis=1,ignore_index=True)
    print('running for combinations: ',count)


    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y,random_state=rr, shuffle=True, test_size=0.3,stratify=data_Y)
    #x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,random_state=rr, shuffle=True, test_size=0.2, stratify=Y_train)
    

    model_XGBoost = XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=121)
    model_XGBoost.fit(X_train,Y_train) #,epochs=no_epochs,verbose=0, callbacks =callbacks_list, shuffle=False, batch_size= 32, steps_per_epoch=len(x_train)//32,validation_batch_size = 32, validation_steps= len(x_val)//32, validation_data=(x_val,y_val))
    probas_2=model_XGBoost.predict_proba(X_test)
    probas_2 = probas_2[:,1]
    probas_XGB.append(probas_2)
    Y_grand.append(Y_test)
    
    #evaluate performance
    fpr_, tpr_, thresholds_ = roc_curve(Y_test,probas_2)
    gene_precision_ = average_precision_score(Y_test,probas_2)
    gene_roc_auc_ = auc(fpr_,tpr_)
    if gene_roc_auc_ < 0.5:
        gene_roc_auc_ = 1 - gene_roc_auc_

    #optimal_idx = np.argmax(tpr_ANN - fpr_ANN)
    optimal_threshold = 0.5 #thresholds_ANN[optimal_idx]  # 0.5 
    
    probas_thresh=(probas_2 - optimal_threshold + 0.5)
    
    precision_t = precision_score(Y_test,probas_thresh.round(), average='binary', labels=[1])
    recall_t=recall_score(Y_test,probas_thresh.round(), average='binary', labels=[1])
    f1_score_t=f1_score(Y_test,probas_thresh.round(), average='binary', labels=[1])
    
    #store results
    auprc_array.append(gene_precision_)
    auc_array.append(gene_roc_auc_)
    avg_precision.append(precision_t)
    avg_recall.append(recall_t)
    avg_f1.append(f1_score_t)
    avg_MCC.append(matthews_corrcoef(Y_test,probas_thresh.round()))
    
    # summarize history for accuracy
    plt.plot(auc_array)
    plt.plot(auprc_array)
    plt.title('model performance')
    plt.ylabel('Performance')
    plt.xlabel('Models')
    plt.legend(['AUC', 'AUPRC'], loc='upper left')
    plt.show()
    
    count +=1

#create results dataframe    
results_combinations =pd.DataFrame(combination_names)
results_combinations['AUC']=auc_array
results_combinations['AUPRC']=auprc_array
results_combinations['F1']=avg_f1
results_combinations['Precision']=avg_precision
results_combinations['Recall']=avg_recall
results_combinations['MCC']=avg_MCC


#sort results based on AUC
results_combinations_sorted = results_combinations.sort_values('AUC', ascending=False)
results_combinations_sorted.to_excel('results_combinations_sorted_%s.xlsx'%(str(30)))

results_idx = results_combinations_sorted.index # use this for best over all

# find clusters in result
X_cluster = np.array(auc_array)
km = ckwrap.ckmeans(X_cluster,5)
buckets = [[],[],[],[],[]]
buckets_idx = [[],[],[],[],[]]

for i in range(len(X_cluster)):
    buckets[km.labels[i]].append(X_cluster[i])
    buckets_idx[km.labels[i]].append(np.where(X_cluster==X_cluster[i])[0])
print(max(buckets[0]), min(buckets[0]))
print(max(buckets[1]), min(buckets[1]))
print(max(buckets[2]), min(buckets[2]))
print(max(buckets[3]), min(buckets[3]))
print(max(buckets[4]), min(buckets[4]))


# export results of the top performers in the 5 clusters, use index printed find cluster code section


dict_results={}

# dict_results['preds_0']=np.ravel(probas_XGB[results_idx[0]])    (boundaries for alpha=100)
# dict_results['preds_1']=np.ravel(probas_XGB[results_idx[34]])
# dict_results['preds_2']=np.ravel(probas_XGB[results_idx[68]])
# dict_results['preds_3']=np.ravel(probas_XGB[results_idx[91]])
# dict_results['preds_4']=np.ravel(probas_XGB[results_idx[112]])

dict_results['preds_0']=np.ravel(probas_XGB[results_idx[0]])      #(boundaries for alpha=30)
dict_results['preds_1']=np.ravel(probas_XGB[results_idx[64]])
dict_results['preds_2']=np.ravel(probas_XGB[results_idx[82]])
dict_results['preds_3']=np.ravel(probas_XGB[results_idx[96]])
dict_results['preds_4']=np.ravel(probas_XGB[results_idx[108]])

dict_results['Y_test']=np.ravel(Y_test)

filename = 'plot_test_%s_combination_clusters5.mat'%(str(30))

sio.savemat(filename,dict_results)




    
