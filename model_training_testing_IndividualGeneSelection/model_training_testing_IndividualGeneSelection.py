"""
Code to train XGBoost and test the model performance with and without feature 
selection from individual features (generates data for Table 7)

"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import gc
gc.disable()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import scipy.io as sio
from xgboost import XGBClassifier
from sklearn.utils import resample

# set some simulation parameters
num_bootsrap = 100 # Number of bootsrap sampling
no_features=25  # Number of most informative features to select

#feature selection
def feat_sel(X,Y):
    fs = SelectKBest(score_func=f_classif, k=no_features)  #no_features
    fs.fit_transform(X,Y)
    X_ind = fs.get_support(indices=True)
    see=fs.scores_
    
    return X_ind, see

# store results
preds_FS = [] 
preds_nFS = []
Y_grand = []
no_features_sel =[]


##import and preprocess data
s_ratio= 1    # set the value of alpha (0.3 and 1 reported in paper), the cut off percent for selection the top and least expressed genes 
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

#remove some redundant and constant features
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
    y_pred = model_XGBoost.predict_proba(X_test)[:,1]
    preds_nFS.append(y_pred)
 
    # Features selection   
    X_ind,feat_scores= feat_sel(X_train,Y_train)   # Features selection basted on F-scores
    X_train = X_train[:,X_ind]
    X_test = X_test[:,X_ind]
      
    # Re-train model with selected features
    selection_model = XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=121)
    selection_model.fit(X_train, Y_train)
    
    # Evaluate  retrained model
    y_pred_s = selection_model.predict_proba(X_test)[:,1]
    preds_FS.append(y_pred_s)
    
    #store bootsrap Y samples
    Y_grand.append(Y_test)
    
preds_FS = np.ravel(preds_FS)
preds_nFS = np.ravel(preds_nFS)
Y_grand = np.ravel(Y_grand)
            
fpr_, tpr_, thresholds_ = roc_curve(Y_grand,preds_nFS)
gene_precision_ = average_precision_score(Y_grand,preds_nFS)
gene_roc_auc_ = auc(fpr_,tpr_)
if gene_roc_auc_ < 0.5:
    gene_roc_auc_ = 1 - gene_roc_auc_
                          
auprc_array_nFS = gene_precision_
auc_array_nFS = gene_roc_auc_

fpr_, tpr_, thresholds_ = roc_curve(Y_grand,preds_FS)
gene_precision_ = average_precision_score(Y_grand,preds_FS)
gene_roc_auc_ = auc(fpr_,tpr_)
if gene_roc_auc_ < 0.5:
    gene_roc_auc_ = 1 - gene_roc_auc_
                          
auprc_array_FS = gene_precision_
auc_array_FS = gene_roc_auc_

print('Model without Feature Selection',auprc_array_nFS,auc_array_nFS)
print('Model with Feature Selection',auprc_array_FS,auc_array_FS)

# export results for plotting on MATLAB
dict_results={}
dict_results['preds_FS']=np.ravel(preds_FS)
dict_results['preds_nFS']=np.ravel(preds_nFS)
dict_results['Y_grand']=np.ravel(Y_grand)

filename = 'plot_test_IndividualFS_Analysis_%s.mat'%(str(int(s_ratio*100)))
#sio.savemat(filename,dict_results)    ###uncomment this line to save the file to disk       