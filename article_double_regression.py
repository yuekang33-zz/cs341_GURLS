#add label to the data, whether a post will double in size
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import pylab
import matplotlib
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

#read in the data
Location  = r'/data/stanford/p_hf/ky_allDistribution'
Location_fan  = r'/data/wechat_data/biz_fans_num.txt'
#Location = r'F:\academic\cs341\wechat\fh\ky_allDistribution'
df_distribution = pd.read_csv(Location,index_col=0,header = None)
df_fancount = pd.read_csv(Location_fan,index_col=None,header=0)
df_distribution.columns = ("appmsgid","bizuin_md5","itemids","cascadeSize")

#merge the two dataframes
df_distribution = pd.merge(df_distribution,df_fancount, on = 'bizuin_md5')

#log regression
step = [5,10,20,40,80]
for i in xrange(len(step)-1):
    df_temp = df_distribution[df_distribution["cascadeSize"].between(step[i],10e6)]
    #df_temp = df_distribution[df_distribution["step"+str(step[i])]==True]
    df_temp = df_temp[np.isfinite(df_temp['fans_num'])]
    X = df_temp["fans_num"].values.tolist()
    X = np.array(X)
    X = X.reshape([len(X),1])
    Y = df_temp["cascadeSize"].between(step[i+1],10e6).tolist()
    #Y = df_temp["step"+str(step[i+1])].tolist()
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
    h=0.02
    logreg = linear_model.LogisticRegression(C=1e5,max_iter=1e3)
    logreg.fit(X_train,y_train)
    print "step"+str(step[i+1])
    print len(X)
    print logreg.score(X_test,y_test)