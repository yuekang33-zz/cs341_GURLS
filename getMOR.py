#feature investigated : 
#   public account: number of fans,length of posts
#   reshare user: average age,sex,education_level,reg_time,friend_count,sendmsg_count,recvmsg_count,snsupload_count,sns_view
#   reshare user's position in the cascade
#   structual: number of first layer users, 
#   more user features
#model:
#   logistic regression    
#   5 fold cross validation added
#   correlation coefficient
#   tuning the model
import pandas as pd
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
import sys
import pylab
import matplotlib
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

LocationReshare = r'/data/wechat_data/article_reshare'
df_reshare = pd.read_csv(LocationReshare,index_col = None, header = 0, sep = "\t")

# Means of reshare
# sns_: 0
# fav_: 1
# msg_: 2

# store cur_publishid in a separate dataframe
df_reshare_mor = pd.DataFrame()
df_reshare_mor['appmsgid'] = df_reshare['appmsgid']
df_reshare_mor['bizuin_md5'] = df_reshare['bizuin_md5']
df_reshare_mor['itemidx'] = df_reshare['itemidx']

#print df_reshare_mor.iloc[0]['cur_publishid']

# splice cur_publishid
#foo = lambda x: pd.Series([i for i in reversed(x.split('_',1))])
foo = lambda x: pd.Series(x.split('_',1))
tmp = df_reshare['cur_publishid'].apply(foo)
tmp1 = tmp.ix[:,0:2]
tmp1.columns = ['value', 'meansOfReshare']
#tmp1.drop_duplicates(cols='meansOfReshare',keep = 'first')
#print tmp1

tmp1.loc[tmp1['meansOfReshare'] == 'sns', 'value'] = 0
tmp1.loc[tmp1['meansOfReshare'] == 'fav', 'value'] = 1
tmp1.loc[tmp1['meansOfReshare'] == 'msg', 'value'] = 2
#tmp1.loc[tmp1['meansOfReshare'] == 'bs', 'value'] = 3
#tmp1.loc[tmp1['meansOfReshare'] == 'bshw', 'value'] = 4
#tmp1.loc[tmp1['meansOfReshare'] == 'app', 'value'] = 5

df_reshare_mor['value'] = tmp1['value']
df_reshare_mor['meansOfReshare'] = tmp1['meansOfReshare']

# store result to .csv
filename ="/data/stanford/p_ruixilin/meansOfReshare.csv"
df_reshare_mor.to_csv(filename)














