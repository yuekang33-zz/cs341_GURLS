#add label to the data, whether a post will double in size
#feature investigated : 
#   public account: number of fans,length of posts
#   reshare user: average age,sex,education_level,reg_time,friend_count,sendmsg_count,recvmsg_count,snsupload_count,sns_view
#   reshare user's position in the cascade
#model:
#   logistic regression    
#   5 fold cross validation added
import pandas as pd
import numpy as np 
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
#read in the data
Location  = r'/data/stanford/p_hf/ky_allDistribution'
Location_fan  = r'/data/wechat_data/biz_fans_num.txt'
LocationReshare = r'/data/wechat_data/article_reshare'
LocationUser = r'/data/wechat_data/userattr_sample'
LocationView = r'/data/wechat_data/article_read'
LocationCascade = r'/data/stanford/p_hf/cascadeINFO/cascadeUserLayer_whole.txt'
#public account data
df_distribution = pd.read_csv(Location,index_col=0,header = 0)
df_distribution.columns = ("appmsgid","bizuin_md5","itemidx","cascadeSize")#3 article id are objects, because header mixed up into the data as a row
df_fancount = pd.read_csv(Location_fan,index_col=None,header=0)#bizuin_md5 only 
df_reshare = pd.read_csv(LocationReshare,index_col = None, header = 0, sep = "\t")#app and item are int64

#public account view data
df_view = pd.read_csv(LocationView, sep = "\t", header = None)#app and item are int64
df_view.columns = ["fdate_cd", "bizuin_md5", "appmsgid", "itemidx", "title", "url", "len", "posttime", "uin_md5", "read_timestamp", "read_scene","platform"];
#get the reshare timepost ready
df_reshare_timestamp = pd.DataFrame()
df_reshare_timestamp['appmsgid'] = df_reshare['appmsgid']#.astype(int)
df_reshare_timestamp['bizuin_md5'] = df_reshare['bizuin_md5']
df_reshare_timestamp['itemidx'] = df_reshare['itemidx']#.astype(int)
df_reshare_timestamp['reshare_timestamp'] = df_reshare['reshare_timestamp']

"""
#change data type for merging, might be unnecessary
df_reshare['appmsgid'] = df_reshare['appmsgid'].astype(object)
df_reshare['itemidx'] = df_reshare['itemidx'].astype(object)
"""

#user data
df_user = pd.read_csv(LocationUser,header = None,sep = "\t", names = ['uin_md5','age','sex','education_level','reg_time','friend_count','sendmsg_count','recvmsg_count','snsupload_count','sns_view'])

#cascade data
df_cascade = pd.read_csv(LocationCascade,index_col = None, header = 0)#app and item are int64

#merge distribution and fancount
df_distribution = pd.merge(df_distribution,df_fancount, on = 'bizuin_md5')

#merge cascade info by article id and user id to reshare
df_reshare = pd.merge(df_reshare,df_cascade,how = "left", on = ['appmsgid','bizuin_md5','itemidx','uin_md5'])
#log regression
step = [5,10,20,40,80,160,320,640,1280,2560,5120,10240]
for i in xrange(len(step)-1):
    #get user feature groupby  article id and merge with reshare
    df_reshare = df_reshare.groupby(['bizuin_md5','appmsgid','itemidx'])
    df_reshare = df_reshare.apply(pd.DataFrame.sort,'reshare_timestamp',ascending  = True)
    df_reshare_drop = df_reshare.groupby(level = ['bizuin_md5','appmsgid','itemidx']).head(step[i])
    df_reshare_drop = pd.merge(df_reshare_drop,df_user, how = 'inner', on = 'uin_md5')
    df_reshare_user = df_reshare_drop.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg(
        {'age':'mean','sex':'mean','education_level':'mean',
        'friend_count':'mean','sendmsg_count':'mean',
        'recvmsg_count':'mean','snsupload_count':'mean','sns_view':'mean','cascadeLayer':'sum'})
         
    #get number of views group by id    
    df_reshare_drop = df_reshare_timestamp.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).apply(pd.DataFrame.sort,'reshare_timestamp',ascending  = True).groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).head(step[i])
    df_reshare_step = df_reshare_drop.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg({'reshare_timestamp':'max'})
    df_view_step = pd.merge(df_view,df_reshare_step, how = 'inner',on = ['appmsgid','bizuin_md5','itemidx'])
    df_topview = df_view_step[df_view_step['reshare_timestamp']>=df_view_step['read_timestamp']]
    df_topview_count = df_topview.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg({'read_timestamp':'count'})
    
    #filter out articles below required size
    df_temp = df_distribution[df_distribution["cascadeSize"].between(step[i],10e6)]
    df_temp['appmsgid'] = df_temp['appmsgid'].astype(int)
    df_temp['itemidx'] = df_temp['itemidx'].astype(int)
    df_temp = pd.merge(df_temp,df_reshare_user, on = ['appmsgid','bizuin_md5','itemidx'])
    """
    print "merge with user info",df_temp.head()
    """
    df_topview_count['appmsgid'] = df_topview_count['appmsgid'].astype(int)
    df_topview_count['itemidx'] = df_topview_count['itemidx'].astype(int)
    df_temp = pd.merge(df_temp,df_topview_count, on = ['appmsgid','bizuin_md5','itemidx'])
    """
    print "merge with view info",df_temp.head()  
    """    
    df_temp = df_temp[np.isfinite(df_temp['fans_num'])]
    df_temp = df_temp[np.isfinite(df_temp['friend_count'])]
    df_temp = df_temp[np.isfinite(df_temp['age'])]
    df_temp = df_temp[np.isfinite(df_temp['sex'])]
    df_temp = df_temp[np.isfinite(df_temp['education_level'])]
    df_temp = df_temp[np.isfinite(df_temp['sendmsg_count'])]
    df_temp = df_temp[np.isfinite(df_temp['recvmsg_count'])]
    df_temp = df_temp[np.isfinite(df_temp['snsupload_count'])]
    df_temp = df_temp[np.isfinite(df_temp['sns_view'])]
    df_temp = df_temp[np.isfinite(df_temp['read_timestamp'])]
    df_temp = df_temp[np.isfinite(df_temp['cascadeLayer'])]
    
    df_features = pd.DataFrame()
    df_features['fans_num'] = df_temp['fans_num']
    df_features['friend_count'] = df_temp['friend_count']
    df_features['age'] = df_temp['age']
    df_features['sex'] = df_temp['sex']
    df_features['education_level'] = df_temp['education_level']
    df_features['sendmsg_count'] = df_temp['sendmsg_count']   
    df_features['recvmsg_count'] = df_temp['recvmsg_count']
    df_features['snsupload_count'] = df_temp['snsupload_count']
    df_features['sns_view'] = df_temp['sns_view']   
    df_features['read_count'] = df_temp['read_timestamp'] 
    df_features['len'] = df_view['len'].astype(int)
    df_features['cascadeLayer'] = df_temp['cascadeLayer']
    df_features['read_count_reverse'] = 1/df_temp['read_timestamp'] 
    """
    print "example of features"
    print df_features.head()
    """
    X = df_features.values.tolist()
    X = np.array(X)
    X = X/X.max(axis=0)
    Y = df_temp["cascadeSize"].between(step[i+1],10e6).tolist()
    Y = np.array(Y)
    #five fold cross-validation
    print "length of total data is", len(X)
    score1= []
    score2= []
    kf = KFold(len(X), n_folds = 5, shuffle = True )
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    #X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
        logreg = linear_model.LogisticRegression(C=1e5,max_iter=1e3)#,class_weight = 'balanced')
        logreg.fit(X_train,y_train)
        print "step"+str(step[i+1])
        print "Percentage of positive data: ",float(sum(Y))/len(Y)
        y_pred = logreg.predict(X_test)
        """
        print classification_report(y_pred,y_test)
        """

        print "Accuracy score w/o feature selection: ",logreg.score(X_test,y_test)
        score1.append(logreg.score(X_test,y_test))
        # perform recursive feature selection(backward selectoin)
        rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(y_train, 4),scoring='accuracy')
        #rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(y_train, 4),scoring='roc_auc')
        rfecv.fit(X_train, y_train)

        print("Optimal number of features : %d" % rfecv.n_features_)
        print "rfecv support ",rfecv.support_
        print "rfecv ranking ", rfecv.ranking_

        # retrain with selected features
        masked_X_train = np.asarray(X_train)
        masked_X_train = masked_X_train[:, rfecv.support_]
        masked_X_train = masked_X_train.tolist()

        logreg.fit(masked_X_train,y_train)

        masked_X_test = np.asarray(X_test)
        masked_X_test = masked_X_test[:, rfecv.support_]
        masked_X_test = masked_X_test.tolist()

        print "Accuracy score w/ features selection",logreg.score(masked_X_test,y_test)
        score2.append(logreg.score(masked_X_test,y_test))     
        masked_y_pred = logreg.predict(masked_X_test)
        """
        print classification_report(y_test,masked_y_pred)
        """
    print "Average accuracy score w/o feature selection:", sum(score1)/len(score1)
    print "Average accuracy score w/ feature selection:", sum(score2)/len(score2)

