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

# splice cur_publishid
foo = lambda x: pd.Series(x.split('_',1))
tmp = df_reshare['cur_publishid'].apply(foo)
tmp1 = tmp.ix[:,0:2]
tmp1.columns = ['meansOfReshare', 'value']
df_reshare_mor['is_sns'] = tmp1['meansOfReshare'] == 'sns'
df_reshare_mor['is_fav'] = tmp1['meansOfReshare'] == 'fav'
df_reshare_mor['is_msg'] = tmp1['meansOfReshare'] == 'msg'

# store result to .csv
filename ="/data/stanford/p_ruixilin/meansOfReshare.csv"
df_reshare_mor.to_csv(filename)






