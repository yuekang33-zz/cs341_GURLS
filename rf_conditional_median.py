#given first few reshares of cascades larger than a given size, predict whether it will reach the median size in the group
#feature investigated : 
#   public account: number of fans,length of posts
#   reshare user: average age,sex,education_level,reg_time,friend_count,sendmsg_count,recvmsg_count,snsupload_count,sns_view
#   reshare user's position in the cascade
#   structual: number of first layer users, 
#   more user features
#model:
#   random forest
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
#read in the data
Location  = r'/data/stanford/p_hf/ky_allDistribution'
Location_fan  = r'/data/wechat_data/biz_fans_num.txt'
LocationReshare = r'/data/wechat_data/article_reshare'
LocationUser = r'/data/wechat_data/userattr_sample'
LocationView = r'/data/wechat_data/article_read'
LocationCascade = r'/data/stanford/p_hf/cascadeINFO/cascadeUserLayer_whole.txt'
LocationMOR = r'/data/stanford/p_ruixilin/meansOfReshare.csv'
#public account data
df_distribution = pd.read_csv(Location,index_col=0,header = 0)
df_distribution.columns = ("appmsgid","bizuin_md5","itemidx","cascadeSize")#3 article id are objects, because header mixed up into the data as a row
df_fancount = pd.read_csv(Location_fan,index_col=None,header=0)#bizuin_md5 only 
df_reshare = pd.read_csv(LocationReshare,index_col = None, header = 0, sep = "\t")#app and item are int64
df_MOR = pd.read_csv(LocationMOR,index_col = None, header = 0, sep = ",")
df_reshare['is_sns'] = df_MOR['is_sns']
df_reshare['is_fav'] = df_MOR['is_fav']
df_reshare['is_msg'] = df_MOR['is_msg']
#public account view data
df_view = pd.read_csv(LocationView, sep = "\t", header = None)#app and item are int64
df_view.columns = ["fdate_cd", "bizuin_md5", "appmsgid", "itemidx", "title", "url", "len", "posttime", "uin_md5", "read_timestamp", "read_scene","platform"];
#get the reshare timepost ready
df_reshare_timestamp = pd.DataFrame()
df_reshare_timestamp['appmsgid'] = df_reshare['appmsgid']#.astype(int)
df_reshare_timestamp['bizuin_md5'] = df_reshare['bizuin_md5']
df_reshare_timestamp['itemidx'] = df_reshare['itemidx']#.astype(int)
df_reshare_timestamp['reshare_timestamp'] = df_reshare['reshare_timestamp']
df_reshare_timestamp['posttime'] = df_reshare['posttime']

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
given_size = [80,160,320,640,1280,2560,5120]
step = [5,10,20,40]
#given_size = [80,160,320]
#step = [5,10]
avgscore =[]
numData = []

for i in xrange(len(step)):

    #get user feature groupby  article id and merge with reshare
    df_reshare = df_reshare.groupby(['bizuin_md5','appmsgid','itemidx'])
    df_reshare = df_reshare.apply(pd.DataFrame.sort,'reshare_timestamp',ascending  = True)
    df_reshare_drop = df_reshare.groupby(level = ['bizuin_md5','appmsgid','itemidx']).head(step[i])
    df_reshare_drop = pd.merge(df_reshare_drop,df_user, how = 'inner', on = 'uin_md5')
    
    ####new feature: structual####
    df_reshare_struct = df_reshare_drop.copy()
    df_reshare_struct["firstLayer"] = df_reshare_struct['cascadeLayer']==1
    df_reshare_struct["stdLayer"] = df_reshare_struct['cascadeLayer'].copy()
    df_reshare_struct["avgLayer"] = df_reshare_struct['cascadeLayer'].copy()
    df_reshare_struct["did_leave"] = df_reshare_struct['cascadeLayer'].copy()
    df_reshare_struct.rename(columns={'friend_count':'borderNodes'},inplace = True)
    df_reshare_structural = df_reshare_struct.groupby(['bizuin_md5','appmsgid','itemidx'],as_index = False).agg({'firstLayer':'sum','borderNodes':'sum','stdLayer':'std','avgLayer':'mean','did_leave':'max','is_sns':'sum','is_fav':'sum','is_msg':'sum'})
    ##add number of fans to bordernodes
    df_reshare_structural = pd.merge(df_reshare_structural,df_fancount, how = "left", on = ['bizuin_md5'])
    df_reshare_structural['borderNodes'] = df_reshare_structural['borderNodes']+df_reshare_structural['fans_num']
    df_reshare_structural = df_reshare_structural.drop('fans_num', axis=1)
    ##change did_leave to true false values
    df_reshare_structural['did_leave']  = df_reshare_structural['did_leave']>1
    #print df_reshare_structural.head()
    ####end of structural feature####

    #additional user features
    #get the number of female users among the first k resharers, avg. 'userhood' of k users since registered
    df_reshare_addUser = df_reshare_drop.copy()
    df_reshare_addUser['is_female'] = df_reshare_addUser['sex']==2
    df_reshare_addUser['userhood'] = df_reshare_addUser['reshare_timestamp'] - df_reshare_addUser['reg_time']
    df_reshare_addUser1 = df_reshare_addUser.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg({'is_female':'sum', 'userhood':'mean'})


    #mean, median, std
    df_reshare_user = df_reshare_drop.copy()
    df_reshare_user['age_mean'] = df_reshare_user['age']
    df_reshare_user['age_std'] = df_reshare_user['age']
    df_reshare_user['age_median'] = df_reshare_user['age']

    df_reshare_user['sex_mean'] = df_reshare_user['sex']
    df_reshare_user['sex_std'] = df_reshare_user['sex']
    df_reshare_user['sex_median'] = df_reshare_user['sex']

    df_reshare_user['edu_mean'] = df_reshare_user['education_level']
    df_reshare_user['edu_std'] = df_reshare_user['education_level']
    df_reshare_user['edu_median'] = df_reshare_user['education_level']

    df_reshare_user['friend_mean'] = df_reshare_user['friend_count']
    df_reshare_user['friend_std'] = df_reshare_user['friend_count']
    df_reshare_user['friend_median'] = df_reshare_user['friend_count']

    df_reshare_user['sendmsg_mean'] = df_reshare_user['sendmsg_count']
    df_reshare_user['sendmsg_std'] = df_reshare_user['sendmsg_count']
    df_reshare_user['sendmsg_median'] = df_reshare_user['sendmsg_count']

    df_reshare_user['recvmsg_mean'] = df_reshare_user['recvmsg_count']
    df_reshare_user['recvmsg_std'] = df_reshare_user['recvmsg_count']
    df_reshare_user['recvmsg_median'] = df_reshare_user['recvmsg_count']

    df_reshare_user['snsupload_mean'] = df_reshare_user['snsupload_count']
    df_reshare_user['snsupload_std'] = df_reshare_user['snsupload_count']
    df_reshare_user['snsupload_median'] = df_reshare_user['snsupload_count']  

    df_reshare_user['snsview_mean'] = df_reshare_user['sns_view']
    df_reshare_user['snsview_std'] = df_reshare_user['sns_view']
    df_reshare_user['snsview_median'] = df_reshare_user['sns_view']                       

    df_reshare_user1 = df_reshare_user.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg({ 
        'age_mean': np.mean, 'age_std': np.std, 'age_median':np.median,
        'sex_mean': np.mean, 'sex_std': np.std, 'sex_median':np.median,
        'edu_mean': np.mean, 'edu_std': np.std, 'edu_median':np.median,
        'friend_mean': np.mean, 'friend_std': np.std, 'friend_median':np.median,
        'sendmsg_mean': np.mean, 'sendmsg_std': np.std, 'sendmsg_median':np.median,
        'recvmsg_mean': np.mean, 'recvmsg_std': np.std, 'recvmsg_median':np.median,
        'snsupload_mean': np.mean, 'snsupload_std': np.std, 'snsupload_median':np.median,
        'snsview_mean': np.mean, 'snsview_std': np.std, 'snsview_median':np.median,
        'cascadeLayer':'sum'})
         
    #get number of views group by id    
    df_reshare_drop = df_reshare_timestamp.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).apply(pd.DataFrame.sort,'reshare_timestamp',ascending  = True).groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).head(step[i])
    df_reshare_timestamp_drop = df_reshare_timestamp.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).apply(pd.DataFrame.sort,'reshare_timestamp',ascending  = True).groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).head(step[i])
    df_reshare_timestamp['reshare_timestamp'].astype(int)
    #df_reshare_timestamp.groupby(['appmsgid','bizuin_md5','itemidx'])['reshare_timestamp'].diff()
    df_reshare_timestamp_drop['diffs'] = df_reshare_timestamp_drop.groupby(['appmsgid','bizuin_md5','itemidx'])['reshare_timestamp'].diff()
    df_reshare_timestamp_drop = df_reshare_timestamp_drop.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).filter(lambda x: len(x) >= step[i])
    df_timeGap = df_reshare_timestamp_drop.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).head(step[i]/2).groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).agg({'diffs':'mean'})
    df_timeGap['bottom'] = df_reshare_timestamp_drop.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).tail(step[i]-1-step[i]/2).groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).agg({'diffs':'mean'})['diffs']
    
    df_reshare_step = df_reshare_drop.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg({'reshare_timestamp':'max'})
    df_timeElapsed = df_reshare_timestamp_drop.groupby(['appmsgid','bizuin_md5','itemidx'],as_index = False).agg({'reshare_timestamp':'max', 'posttime': 'min'})
    df_timeElapsed['timeElapsed'] = df_timeElapsed['reshare_timestamp'] - df_timeElapsed['posttime']

    df_view_step = pd.merge(df_view,df_reshare_step, how = 'inner',on = ['appmsgid','bizuin_md5','itemidx'])
    df_topview = df_view_step[df_view_step['reshare_timestamp']>=df_view_step['read_timestamp']]
    df_topview = df_topview.rename(columns = {'reshare_timestamp':'numViews'})
    df_topview_count = df_topview.groupby(['bizuin_md5','appmsgid','itemidx'],
        as_index = False).agg({'numViews':'count','len':'first'})
    df_topview_count['appmsgid'] = df_topview_count['appmsgid'].astype(int)
    df_topview_count['itemidx'] = df_topview_count['itemidx'].astype(int)   
    df_timeElapsed = df_timeElapsed .drop('reshare_timestamp', axis =1)
    df_timeElapsed['posttime_date'] = df_timeElapsed['posttime']
    df_timeElapsed['posttime_date'] = pd.to_datetime(df_timeElapsed['posttime_date'], unit= 's') 
    df_timeElapsed.set_index('posttime_date', inplace=True)
    df_timeElapsed['hour'] = df_timeElapsed.index.hour
    df_timeElapsed = df_timeElapsed .drop('posttime', axis =1)
    df_timeElapsed = df_timeElapsed.reset_index(level=1)
    df_timeElapsed = df_timeElapsed.drop(['posttime_date'], axis=1)

    ####create a 2d list of coefficients
    coefficient_results = []    
    for j in xrange(len(given_size)):
        #filter out articles below required size
        df_temp = df_distribution[df_distribution["cascadeSize"].between(given_size[j],10e6)]
        df_temp['appmsgid'] = df_temp['appmsgid'].astype(int)
        df_temp['itemidx'] = df_temp['itemidx'].astype(int)
        df_temp = pd.merge(df_temp,df_reshare_user1, on = ['appmsgid','bizuin_md5','itemidx'])
        df_temp = pd.merge(df_temp,df_reshare_addUser1, on = ['appmsgid','bizuin_md5','itemidx'])
        
        ####new merge with structure information
        df_temp = pd.merge(df_temp,df_reshare_structural, on = ['appmsgid','bizuin_md5','itemidx'])
        
        """
        print "merge with user info",df_temp.head()
        """
        df_temp = pd.merge(df_temp,df_topview_count, on = ['appmsgid','bizuin_md5','itemidx'])

        df_temp = pd.merge(df_temp,df_timeElapsed, on = ['appmsgid','bizuin_md5','itemidx'])
        df_temp = pd.merge(df_temp,df_timeGap, on = ['appmsgid','bizuin_md5','itemidx'])
        """
        print "merge with view info",df_temp.head()  
        """    
        df_temp = df_temp.dropna(how = "any")

        df_temp = df_temp.drop('appmsgid', axis =1)
        df_temp = df_temp.drop('itemidx', axis =1 )
        df_temp = df_temp.drop('bizuin_md5', axis =1 )
       # df_temp['timeElapsed'] = df_timeElapsed['timeElapsed']
       # df_temp['top'] = df_timeGap['diffs']
       # df_temp['bottom'] = df_timeGap['bottom']
        df_temp['viewspeed'] = df_temp['numViews']/df_temp['timeElapsed']

        median_size = df_temp["cascadeSize"].median()
        print "median size", median_size
        Y = df_temp["cascadeSize"].between(median_size,10e6).tolist()
        Y = np.array(Y)
        df_temp = df_temp.drop('cascadeSize', axis =1 )
        #print "size of sex_median", len(df_temp['sex_median'].values.tolist())
        #print "size of y", len(Y)
        
        """
        for k in df_temp.keys():
            #print "rho (%s, y)"%(k), scipy.stats.pearsonr(df_temp[k].values.tolist(), Y)
            resCoeffx.iloc[i][k]= scipy.stats.pearsonr(df_temp[k].values.tolist(), Y)
        """
        counter_coe = 0;
        for col_name in df_temp.keys():
            #first append the col_names
            if(j==0):
                coefficient_results.append([col_name])
            coefficient_results[counter_coe].append(scipy.stats.pearsonr(df_temp[col_name].values.tolist(), Y)[0])    
            counter_coe+=1
            
        X = df_temp.values.tolist()
        X = np.array(X)
        X = X/X.max(axis=0)
        #five fold cross-validation
        numData.append(len(X))
        score1= []
        score2= []
        kf = KFold(len(X), n_folds = 5, shuffle = True )
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        #X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
            rffit = RandomForestClassifier(n_estimators=100, criterion='gini', 
                max_depth=None, min_samples_split=2, min_samples_leaf=50, 
                min_weight_fraction_leaf=0.0, max_features='sqrt', 
                max_leaf_nodes=None, bootstrap=True, oob_score=True, 
                n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)
            rffit.fit(X_train,y_train)
            """
            print "step"+str(step[i])
            print "Percentage of positive data: ",float(sum(Y))/len(Y)
            """
            y_pred = rffit.predict(X_test)

            #print "Accuracy score w/o feature selection: ",logreg.score(X_test,y_test)
            score1.append(rffit.score(X_test,y_test))
        """    
        print "Average accuracy score w/o feature selection:", sum(score1)/len(score1)
        """
        avgscore.append(sum(score1)/len(score1))
        #print "Average accuracy score w/ feature selection:", sum(score2)/len(score2)
    print "step",step[i]    
    print "average scores are :", avgscore   
    print "number of data: ", numData
    print "coefficents by cascade size: "
    for j in xrange(len(coefficient_results)):
        print coefficient_results[j][0]+str(step[i])+"="+str(coefficient_results[j][1:])
