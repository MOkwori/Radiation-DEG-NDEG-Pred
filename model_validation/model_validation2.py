
"""
Code to validate the five machine learning models.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB


##import and preprocess data
s_ratio= 1    # set the value of alpha, the cut off percent for selection the top and least expressed genes 
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

# generate one-hot-encoded features for tissue
X_tissue = data_X.iloc[:,0]
df= pd.get_dummies(X_tissue)
col_list = df.columns.tolist()
data_X[col_list]=pd.get_dummies(X_tissue)

#remove redundant and constant features
del data_X['tissue']
del data_X['mmu-miR-137-3p']

#normalize, scale and transform features to -0.5 to 0.5 range
data_X.fillna(0, inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data_X =scaler.fit_transform(data_X)
data_X=data_X-0.5

#split data into training and testing set
indices = np.arange(len(data_X))
X_train, X_test, Y_train, Y_test,idx_train, idx_test = train_test_split(data_X, data_Y,indices, random_state=42, shuffle=True, test_size=0.3,stratify=data_Y)

#split training data into train and validation sets
rr= 42 
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train,random_state=rr, shuffle=True, test_size=0.2, stratify=Y_train)

#define cross validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

##Gridsearch ANN
param_grid= [
    {'alpha': [0.001, 0.01, 0.1],
     'hidden_layer_sizes': [(121),(62),(31),(121,121),(62,121),(121,62),(121,121,121),(31,62,121), (121,62,31)],
     'activation': ['logistic', 'tanh', 'relu'],
     }
]
model_ANN = MLPClassifier(random_state=0, max_iter=300, learning_rate_init=0.01,learning_rate='adaptive',early_stopping = True,batch_size=64)
search = GridSearchCV(estimator=model_ANN, param_grid=param_grid,n_jobs = -1, scoring="roc_auc", cv=cv)
search.fit(x_train, y_train)

results_ANN = pd.DataFrame(search.cv_results_)
results_ANN = results_ANN.sort_values(by=["rank_test_score"])
results_ANN = results_ANN.set_index(results_ANN["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("estimator__early_stopping")

##Gridsearch SVC
param_grid = [
    {"C": [0.001, 0.01, 0.1, 1, 10]},
    {"gamma" :[0.001, 0.01, 0.1, 1]},
    {"kernel": ["linear"]},
    {"kernel": ["poly"], "degree": [2,3,4,5]},
    {"kernel": ["rbf"]}
]

model_svc = SVC(random_state=0)
search = GridSearchCV(estimator=model_svc, param_grid=param_grid, n_jobs = -1, scoring="roc_auc", cv=cv)
search.fit(x_train, y_train)

results_svc = pd.DataFrame(search.cv_results_)
results_svc = results_svc.sort_values(by=["rank_test_score"])
results_svc = results_svc.set_index(results_svc["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("kernel")

### Gridsearch XGBoost
param_grid = {
    'max_depth': [5, 10, 15],
    'n_estimators': [31, 62, 121],
    'learning_rate': [0.1, 0.01, 0.05]
}

model_XGBoost = XGBClassifier(objective= 'binary:logistic',nthread=4,seed=42)
search = GridSearchCV(estimator=model_XGBoost, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs = -1,verbose=True)
search.fit(x_train, y_train)

results_xgb = pd.DataFrame(search.cv_results_)
results_xgb = results_xgb.sort_values(by=["rank_test_score"])
results_xgb = results_xgb.set_index(results_xgb["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("n_estimators")


### Gridsearch Random forest
param_grid = {
    'max_depth': [5, 10, 15],
    'max_features': [10, 20, 30],
    'n_estimators': [31, 62, 121]
}

model_RF = RandomForestClassifier()
search = GridSearchCV(estimator=model_RF, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs = -1,verbose=True)
search.fit(x_train, y_train)

results_rf = pd.DataFrame(search.cv_results_)
results_rf = results_rf.sort_values(by=["rank_test_score"])
results_rf = results_rf.set_index(results_rf["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("n_estimators")


num_bootsrap = 100
auc_array_ann=np.zeros((5,num_bootsrap)) #[] ANN, SVC, NaiveBayes, XGBoost, Random Forest
auprc_array_ann = np.zeros((5,num_bootsrap)) #[] ANN, SVC, NaiveBayes, XGBoost, Random Forest

probas_ANN = []
probas_SVC = []
probas_NB = []
probas_XGB = []
probas_RF = []
Y_output = []


#train and compare best performing parameters of five models
for m in range(num_bootsrap):
    print('Running for bootsrap: ',m) 
    
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train,random_state=rr, shuffle=True, test_size=0.2, stratify=Y_train)

    
    #set to training set
    tr_size = int(len(x_train) * 0.8)
    te_size = int(len(x_test) * 0.8)
    
    x_train,y_train = resample(x_train,y_train, n_samples=tr_size)
    x_test,y_test = resample(x_test,y_test, n_samples=te_size)
    
    #ANN
    model = MLPClassifier(activation = "tanh", alpha = 0.001, hidden_layer_sizes= (31, 62, 121), random_state=0, max_iter=300, learning_rate_init=0.01,learning_rate='adaptive',early_stopping = True,batch_size=64)
    model.fit(x_train,y_train)
    probas_2=model.predict(x_test)
    probas_ANN.append(probas_2)
    Y_test_names=[gene_names[i] for i in y_test.index]
    
    # gene_ANN = np.ravel(y_pred_train)  
    fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(y_test,probas_2)
    gene_precision_ANN = average_precision_score(y_test,probas_2)
    gene_roc_auc_ANN = auc(fpr_ANN,tpr_ANN)
    if gene_roc_auc_ANN < 0.5:
        gene_roc_auc_ANN = 1 - gene_roc_auc_ANN
        
    
    auprc_array_ann[0,m] = gene_precision_ANN
    auc_array_ann[0,m] = gene_roc_auc_ANN
   
    # Support Vector Classifier
    model_SVC=SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='scale', kernel='linear',
              max_iter=-1, probability=True, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    
    model_SVC.fit(x_train,y_train)
    
    probas_2=model_SVC.predict_proba(x_test)
    probas_2 = probas_2[:,1]
    probas_SVC.append(probas_2)
    
    fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(y_test,probas_2)
    gene_precision_ANN = average_precision_score(y_test,probas_2)
    gene_roc_auc_ANN = auc(fpr_ANN,tpr_ANN)
    if gene_roc_auc_ANN < 0.5:
        gene_roc_auc_ANN = 1 - gene_roc_auc_ANN
                          
    auprc_array_ann[1,m] = gene_precision_ANN
    auc_array_ann[1,m] = gene_roc_auc_ANN
    
    #Naive Bayes Classifier
    model_NaiveBayes = GaussianNB()
    model_NaiveBayes.fit(x_train,y_train)
    probas_2=model_NaiveBayes.predict_proba(x_test)
    probas_2 = probas_2[:,1]
    probas_NB.append(probas_2)
    
    fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(y_test,probas_2)
    gene_precision_ANN = average_precision_score(y_test,probas_2)
    gene_roc_auc_ANN = auc(fpr_ANN,tpr_ANN)
    if gene_roc_auc_ANN < 0.5:
        gene_roc_auc_ANN = 1 - gene_roc_auc_ANN
                          
    auprc_array_ann[2,m] = gene_precision_ANN
    auc_array_ann[2,m] = gene_roc_auc_ANN
    
    #XGBoost model
    model_XGBoost = XGBClassifier(learning_rate=0.01,max_depth=2,n_estimators=242)
    model_XGBoost.fit(x_train,y_train)
    probas_2=model_XGBoost.predict_proba(x_test)
    probas_2 = probas_2[:,1]
    probas_XGB.append(probas_2)
    
    fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(y_test,probas_2)
    gene_precision_ANN = average_precision_score(y_test,probas_2)
    gene_roc_auc_ANN = auc(fpr_ANN,tpr_ANN)
    if gene_roc_auc_ANN < 0.5:
        gene_roc_auc_ANN = 1 - gene_roc_auc_ANN
                          
    auprc_array_ann[3,m] = gene_precision_ANN
    auc_array_ann[3,m] = gene_roc_auc_ANN
    
    ### Random forest classiffier
    model_RF = RandomForestClassifier(max_depth=5, max_features=30, n_estimators=121, random_state=0)
    model_RF.fit(x_train,y_train)
    probas_2=model_RF.predict_proba(x_test)
    probas_2 = probas_2[:,1]
    probas_RF.append(probas_2)
    
    fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(y_test,probas_2)
    gene_precision_ANN = average_precision_score(y_test,probas_2)
    gene_roc_auc_ANN = auc(fpr_ANN,tpr_ANN)
    if gene_roc_auc_ANN < 0.5:
        gene_roc_auc_ANN = 1 - gene_roc_auc_ANN
                          
    auprc_array_ann[4,m] = gene_precision_ANN
    auc_array_ann[4,m] = gene_roc_auc_ANN
    
    Y_output.append(y_test)

#plot box plots
import matplotlib.pyplot as plt    
fig = plt.figure()
ax = fig.add_subplot(111)
auc2 =auc_array_ann.transpose()
plt.boxplot(auc2)
names = ['ANN', 'SVC', 'NaiveBayes', 'XGBoost', 'Random Forest']
ax.set_xticklabels(names)
plt.show()
#plt.savefig('auc_comp_01.pdf')  #comment plt.show() and uncomment this line to save plot

fig = plt.figure()
#fig.suptitle('Algorithm Comparison: AUPRC')
ax = fig.add_subplot(111)
auc2 =auprc_array_ann.transpose()
plt.boxplot(auc2)
names = ['ANN', 'SVC', 'NaiveBayes', 'XGBoost', 'Random Forest']
ax.set_xticklabels(names)
plt.show()
#plt.savefig('auprc_comp_01.pdf') #comment plt.show() and uncomment this line to save plot

#run pairwise comparison of model's AUC 
from itertools import combinations
from math import factorial
import numpy as np
from scipy.stats import t

n_train=len(X_train)
n_test=len(X_test)


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

model_scores = pd.DataFrame(auc_array_ann, index = ['ANN','SVC', 'NaiveBayes', 'XGBoost', 'Random Forest'])

n_comparisons = factorial(len(model_scores)) / (factorial(2) * factorial(len(model_scores) - 2))
pairwise_t_test = []

n = len(model_scores[0]) # number of test sets
df = n - 1


for model_i, model_k in combinations(range(len(model_scores)), 2):
    model_i_scores = model_scores.iloc[model_i].values
    model_k_scores = model_scores.iloc[model_k].values
    differences = model_i_scores - model_k_scores
    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    p_val *= n_comparisons  # implement Bonferroni correction
    # Bonferroni can output p-values higher than 1
    p_val = 1 if p_val > 1 else p_val
    pairwise_t_test.append(
        [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
    )

pairwise_comp_df = pd.DataFrame(pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]).round(3)

