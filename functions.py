import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,roc_auc_score,roc_curve,confusion_matrix


def ceil(x):
    return np.ceil(x)

def standardize(df,variables):
    for k in variables:
        if k == 'targets':
            continue
        else:
            df[k] = (df[k]-np.mean(df[k]))/np.std(df[k]) 
            
    return df

def qq_plot(df,variables):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]
    
    i=0
    df = standardize(df,variables)
    for field in df[variables]:
        stats.probplot(df[field], dist="norm", plot=ax[i])
        ax[i].set_title("Q-Q plot: "+field)
        i+=1
    plt.show()     
    return df

def boxplots(df, var):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    fig.suptitle('boxplots')

    sns.boxplot(ax=axes[0, 0], x='targets', y=var[0], data=df)
    sns.boxplot(ax=axes[0, 1], x='targets', y=var[1], data=df)
    sns.boxplot(ax=axes[1, 0], x='targets', y=var[2], data=df)
    sns.boxplot(ax=axes[1, 1], x='targets', y=var[3], data=df)

def log_transform(df,variables):
    for v in variables:
        print(v)
        print('Skewness (before):',df[v].skew())
        df[v] = np.log10(df[v])
        print('Skewness (after):',df[v].skew(),'\n')
        
    return df

def plot_corr(data,var):

    c = list(data.copy().columns)
    c.remove(var)
    d = data[c]
    
    corr = d.corr()
    f,ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    plt.show()

def find_high_corr_pairs(df):
    corr = df.corr()
    corrlist = []
    for c in corr:
        corr_target = corr[c]
        relevant_features = dict(corr_target[corr_target>0.6])
        del relevant_features[c]

        if len(relevant_features)>0:
            for k in relevant_features:
                s = set([c,k,relevant_features[k]])
                if s not in corrlist:
                    corrlist.append(s)
    return corrlist

def calculate_vif(dataX):
    vif_data = pd.DataFrame()
    vif_data["feature"] = dataX.columns

    vif_data["VIF"] = [variance_inflation_factor(dataX.values, i) for i in range(len(dataX.columns))]

    return vif_data[vif_data['VIF']>10]

def drop(df,var):
    return df.copy().drop(var,axis=1)
