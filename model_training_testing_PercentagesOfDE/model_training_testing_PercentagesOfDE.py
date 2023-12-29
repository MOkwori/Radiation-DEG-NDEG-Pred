"""
Code to train XGBoost and test the performance on percentages of DE 
(generates data for Table 6 and Figure 3)

"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import gc
gc.disable()
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import scipy.io as sio


##import and preprocess data
s_ratio= 1    # set the value of alpha to 0.1, 0.3, 0.5, 0.7, and 1. This is the cut off percent for selection the top and least expressed genes 
gene_df = pickle.load(open( "full_engineered_dataset.p", "rb"))    # loads full engineered dataset
gene_df=gene_df.reset_index(drop=True)

gene_df_ndeg = gene_df[gene_df['Output']==0]
gene_df_deg = gene_df[gene_df['Output']==1]

gene_df_deg2=gene_df_deg.sort_values('Adj. p-value')
gene_df_ndeg2=gene_df_ndeg.sort_values('Adj. p-value')

gene_df_ndeg = gene_df_ndeg2
gene_df_deg = gene_df_deg2

#sel perventages of the most and least DE genes
sel_ind=list(gene_df_deg2[:int(len(gene_df_deg)*s_ratio)].index)
sel_ind.extend(list(gene_df_ndeg2[-int(len(gene_df_deg)*s_ratio):].index))
gene_df_0=gene_df.iloc[sel_ind,:]

# drop columns not used in training                             
rem_col =['Fold Change (Log2)','Adj. p-value','glds', 'Gene Symbol', 'abs_FC', 'NDEG', 'DEG', 'Drop']
gene_df_proc=gene_df_0.drop(rem_col, axis=1) 

gene_names = gene_df_0['Gene Symbol']    # track names of genes

#split dataset into Input and Output
data=gene_df_proc
data_Y = data.iloc[:,116]
data_X = data.iloc[:,0:116]
feature_list = list(data.columns)

# generate one-hot-encoded features for tissue
X_tissue = data_X.iloc[:,0]
df= pd.get_dummies(X_tissue)
col_list = df.columns.tolist()
feature_list.extend(col_list)
data_X[col_list]=pd.get_dummies(X_tissue)

#remove redundant and constant features
del data_X['tissue']
feature_list.remove('tissue')
feature_list.remove('Output')
del data_X['mmu-miR-196a-2-3p']   
feature_list.remove('mmu-miR-196a-2-3p')

#normalize some columns
data_X.fillna(0, inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data_X =scaler.fit_transform(data_X)
data_X=data_X-0.5

#split data into training and testing set
indices = np.arange(len(data_X))
X_train, X_test, Y_train, Y_test,idx_train, idx_test = train_test_split(data_X, data_Y,indices, random_state=42, shuffle=True, test_size=0.3,stratify=data_Y)


num_bootsrap = 100

# Generate arrays for performance metrics
auc_array_=np.zeros((1,num_bootsrap)) 
auprc_array_ = np.zeros((1,num_bootsrap)) 
avg_precision = np.zeros((1,num_bootsrap))
avg_recall= np.zeros((1,num_bootsrap))
avg_f1 = np.zeros((1,num_bootsrap))
avg_MCC= np.zeros((1,num_bootsrap))
probas_XGB = []
Y_output = []


# run boostrap training and testing     
for m in range(num_bootsrap):
    print('Running bootstrap: ',m)
    
    #split data into training and testing set (keep random seed constant)
    indices = np.arange(len(data_X))
    X_train, X_test, Y_train, Y_test,idx_train, idx_test = train_test_split(data_X, data_Y,indices, random_state=42, shuffle=True, test_size=0.3,stratify=data_Y)
  
    #Boostrap spliting training and testing set
    tr_size = int(len(X_train) * 0.8)
    te_size = int(len(X_test) * 0.8)

    X_train,Y_train = resample(X_train,Y_train, n_samples=tr_size, stratify =Y_train)
    X_test,Y_test = resample(X_test,Y_test, n_samples=te_size, stratify =Y_test)
  
    #define and train XGBoost model using best parameters from validation
    model_XGBoost = XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=121) ##max_depth=2,n_estimators=242)
    model_XGBoost.fit(X_train,Y_train)
    
    #use trained model to predict test set and evaluate performance
    probas_2 = model_XGBoost.predict_proba(X_test)[:,1]
        
    fpr_, tpr_, thresholds_ = roc_curve(Y_test,probas_2)
    gene_precision_ = average_precision_score(Y_test,probas_2)
    gene_roc_auc_ = auc(fpr_,tpr_)
    if gene_roc_auc_ < 0.5:
        gene_roc_auc_ = 1 - gene_roc_auc_
                          
    auprc_array_[0,m] = gene_precision_
    auc_array_[0,m] = gene_roc_auc_
    
    #optimal_idx = np.argmax(tpr_ANN - fpr_ANN)
    optimal_threshold = 0.5    #thresholds_ANN[optimal_idx]
    
    probas_thresh=(probas_2 - optimal_threshold + 0.5)
    
    precision_t = precision_score(Y_test,probas_thresh.round(), average='binary', labels=[1])
    recall_t=recall_score(Y_test,probas_thresh.round(), average='binary', labels=[1])
    f1_score_t=f1_score(Y_test,probas_thresh.round(), average='binary', labels=[1])
    
    
    avg_precision[0,m] = precision_t
    avg_recall[0,m] = recall_t
    avg_f1[0,m] = f1_score_t
    avg_MCC[0,m] = matthews_corrcoef(Y_test,probas_thresh.round())

    #concatenate predictions and Output over boostrapping
    probas_XGB.append(probas_2)       
    Y_output.append(Y_test)
        
#calculate average (with confidence intervals) of performance over the boostrap splitting and samlpling
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower2 = max(0.0, np.percentile(auc_array_[0,:], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper2 = min(1.0, np.percentile(auc_array_[0,:], p))
print('AUC:   ', np.mean(auc_array_[0,:]),lower2, upper2, np.std(auc_array_[0,:]))


p = ((1.0-alpha)/2.0) * 100
lower1 = max(0.0, np.percentile(auprc_array_[0,:], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper1 = min(1.0, np.percentile(auprc_array_[0,:], p))
print('AUPRC:   ', np.mean(auprc_array_[0,:]),lower1, upper1, np.std(auprc_array_[0,:]))

p = ((1.0-alpha)/2.0) * 100
lower3 = max(0.0, np.percentile(avg_f1[0,:], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper3 = min(1.0, np.percentile(avg_f1[0,:], p))
print('F1:   ', np.mean(avg_f1[0,:]),lower3, upper3, np.std(avg_f1[0,:]))

p = ((1.0-alpha)/2.0) * 100
lower4 = max(0.0, np.percentile(avg_recall[0,:], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper4 = min(1.0, np.percentile(avg_recall[0,:], p))
print('Recal:   ', np.mean(avg_recall[0,:]),lower4, upper4, np.std(avg_recall[0,:]))

p = ((1.0-alpha)/2.0) * 100
lower5 = max(0.0, np.percentile(avg_precision[0,:], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper5 = min(1.0, np.percentile(avg_precision[0,:], p))
print('Precision:   ', np.mean(avg_precision[0,:]),lower5, upper5, np.std(avg_precision[0,:]))

p = ((1.0-alpha)/2.0) * 100
lower6 = max(0.0, np.percentile(avg_MCC[0,:], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper6 = min(1.0, np.percentile(avg_MCC[0,:], p))
print('MCC:   ', np.mean(avg_MCC[0,:]),lower6, upper6, np.std(avg_MCC[0,:]))

# export results for plotting on MATLAB
dict_results={}
dict_results['auprc_array_']=np.ravel(auprc_array_)
dict_results['auc_array_']=np.ravel(auc_array_)
dict_results['avg_f1']=np.ravel(avg_f1)
dict_results['avg_recall']=np.ravel(avg_recall)
dict_results['avg_precision']=np.ravel(avg_precision)
dict_results['avg_MCC']=np.ravel(avg_MCC)
dict_results['preds']=np.ravel(probas_2)
dict_results['Y_result']=np.ravel(Y_test)

filename = 'plot_test_PercentageDE_Analysis%s.mat'%(str(int(s_ratio*100)))
#sio.savemat(filename,dict_results)    ###uncomment this line to save the file to disk