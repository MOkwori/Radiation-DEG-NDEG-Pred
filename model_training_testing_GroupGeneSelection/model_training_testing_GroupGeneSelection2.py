# -*- coding: utf-8 -*-
"""
Code to train XGBoost and test the model performance of feature 
selection from combinations of groups of similar features 
(generates data for Table 9 and Figure 5)

"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import gc
gc.disable()
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import itertools
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


#store performance
auc_array = [] 
auprc_array = []
avg_precision = []
avg_recall= []
avg_f1 = []
avg_MCC = []
probas_XGB_ext = []
Y_grand_ext = []


#set simulation parameters
random_state_seed= 42 
num_bootsrap = 100     

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
    print('Running for combinations: ',count)
    
    probas_XGB_int = []
    Y_grand_int = []

    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y,random_state=random_state_seed, shuffle=True, test_size=0.3,stratify=data_Y)
      
    for m in range(num_bootsrap):
        print('Running for bootsrap: ',m)
        
        tr_size = int(len(X_train) * 0.8)
        te_size = int(len(X_test) * 0.8)

        X_train_sampled,Y_train_sampled = resample(X_train,Y_train, n_samples=tr_size,stratify =Y_train)
        X_test_sampled,Y_test_sampled = resample(X_test,Y_test, n_samples=te_size, stratify =Y_test)

        model_XGBoost = XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=121)
        model_XGBoost.fit(X_train_sampled,Y_train_sampled) #,epochs=no_epochs,verbose=0, callbacks =callbacks_list, shuffle=False, batch_size= 32, steps_per_epoch=len(x_train)//32,validation_batch_size = 32, validation_steps= len(x_val)//32, validation_data=(x_val,y_val))
        probas_2=model_XGBoost.predict_proba(X_test_sampled)
        probas_2 = probas_2[:,1]
        probas_XGB_int.append(probas_2)
        Y_grand_int.append(Y_test_sampled)
    
    Y_grand_int = np.ravel(Y_grand_int)
    probas_XGB_int = np.ravel(probas_XGB_int)
    
    Y_grand_ext.append(Y_grand_int)
    probas_XGB_ext.append(probas_XGB_int)
    
    fpr_, tpr_, thresholds_ = roc_curve(Y_grand_int,probas_XGB_int)
    gene_precision_ = average_precision_score(Y_grand_int,probas_XGB_int)
    gene_roc_auc_ = auc(fpr_,tpr_)
    if gene_roc_auc_ < 0.5:
        gene_roc_auc_ = 1 - gene_roc_auc_
                          
    auprc_array.append(gene_precision_)
    auc_array.append(gene_roc_auc_)
    
    #optimal_idx = np.argmax(tpr_ANN - fpr_ANN)
    optimal_threshold = 0.5 #thresholds_ANN[optimal_idx]  # 0.5 
    
    probas_thresh=(probas_XGB_int - optimal_threshold + 0.5)
    
    precision_t = precision_score(Y_grand_int,probas_thresh.round(), average='binary', labels=[1])
    recall_t=recall_score(Y_grand_int,probas_thresh.round(), average='binary', labels=[1])
    f1_score_t=f1_score(Y_grand_int,probas_thresh.round(), average='binary', labels=[1])
    
    
    avg_precision.append(precision_t)
    avg_recall.append(recall_t)
    avg_f1.append(f1_score_t)
    avg_MCC.append(matthews_corrcoef(Y_grand_int,probas_thresh.round()))
    
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
results_combinations_sorted.to_excel('results_combinations_sorted_bootsrap2%s.xlsx'%(str(100)))
results_idx = results_combinations_sorted.index # use this for best over all

# find clusters in result
X_cluster = np.array(auc_array)
km = ckwrap.ckmeans(X_cluster,5)
buckets = [[],[],[],[],[]]
buckets_idx = [[],[],[],[],[]]

for i in range(len(X_cluster)):
    buckets[km.labels[i]].append(X_cluster[i])
    buckets_idx[km.labels[i]].append(np.where(X_cluster==X_cluster[i])[0])
print(max(buckets[0]), min(buckets[0]), results_combinations.index[results_combinations['AUC']==max(buckets[0])].tolist())
print(max(buckets[1]), min(buckets[1]), results_combinations.index[results_combinations['AUC']==max(buckets[1])].tolist())
print(max(buckets[2]), min(buckets[2]), results_combinations.index[results_combinations['AUC']==max(buckets[2])].tolist())
print(max(buckets[3]), min(buckets[3]), results_combinations.index[results_combinations['AUC']==max(buckets[3])].tolist())
print(max(buckets[4]), min(buckets[4]), results_combinations.index[results_combinations['AUC']==max(buckets[4])].tolist())


# export results of the top performers in the 5 clusters, use index printed find cluster code section
dict_results={}
dict_results['preds_0']=np.ravel(probas_XGB_ext[122])      #(set the index to the values printed out in lines 254 - 8)
dict_results['preds_1']=np.ravel(probas_XGB_ext[64])
dict_results['preds_2']=np.ravel(probas_XGB_ext[43])
dict_results['preds_3']=np.ravel(probas_XGB_ext[14])
dict_results['preds_4']=np.ravel(probas_XGB_ext[40])

dict_results['Y_test_0']=np.ravel(Y_grand_ext[122])
dict_results['Y_test_1']=np.ravel(Y_grand_ext[64])
dict_results['Y_test_2']=np.ravel(Y_grand_ext[43])
dict_results['Y_test_3']=np.ravel(Y_grand_ext[14])
dict_results['Y_test_4']=np.ravel(Y_grand_ext[40])

filename = 'plot_test_%s_combination_clusters5_bootsrap.mat'%(str(100))
#sio.savemat(filename,dict_results)     #uncomment this line to save to disk
