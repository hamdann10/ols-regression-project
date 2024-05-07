import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm 

from data_processing import split_data

def correlation_among_numeric_features(df,cols):
    numeric_col=df[cols]
    corr=numeric_col.corr()
    corr_feature=set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j])>0.8:
                colname=corr.columns[i]
                corr_feature.add(colname)
    return corr_feature

def lr_model(x_train,y_train):
    x_train_with_intercept=sm.add_constant(x_train)
    lr=sm.OLS(y_train,x_train_with_intercept).fit()
    return lr

def identify_significant_vars(lr,p_value_threshold=0.05):
    print(lr.pvalues)
    print(lr.rsquared)
    print(lr.rsquared_adj)
    significant_vars=[var for var in lr.pvalues.keys()if lr.pvalues[var]< p_value_threshold]
    return significant_vars

if __name__=="__main__":
    capped_data=pd.read_csv("data/capped_data.csv")
    # print(capped_data.shape)
    corr_feature=correlation_among_numeric_features(capped_data,capped_data.columns)
    # print(corr_feature)

highy_corr_cols=[
    'lower_bound',
    'state_ District of Columbia',
    'upper_bound',
    'median',
    'PctEmpPrivCoverage', 
    'PctPublicCoverageAlone',
    'PctPrivateCoverage', 
    'popEst2015', 
    'PctPrivateCoverageAlone',
    'MedianAgeFemale', 
    'PctMarriedHouseholds', 
    'povertyPercent', 
    'MedianAgeMale'
]
cols=[col for col in capped_data.columns if col not in highy_corr_cols]
len(cols)
x_train,y_train,y_test=split_data(capped_data[cols],"TARGET_deathRate")
lr=lr_model(x_train,y_train)
summary=lr.summary()
print(summary)

significant_vars=identify_significant_vars(lr)
print(len(significant_vars))

# significant_vars.remove("const")
x_train=sm.add_constant(x_train)
lr=lr_model(x_train[significant_vars],y_train)
summary=lr.summary()
summary