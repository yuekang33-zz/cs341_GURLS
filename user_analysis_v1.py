#to count the number of reshares of female and male user
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import pylab
import matplotlib

#read data
#Location = r'/data/wechat_data/userattr_sample_reshareuser_only'
#df = pd.read_csv(Location,header = 0,sep = "\t")
Location = r'/data/wechat_data/userattr_sample'
df = pd.read_csv(Location,header = None,sep = "\t", names = ['uin_md5','age','sex','education_level','reg_time','friend_count','sendmsg_count','recvmsg_count','snsupload_count','sns_view'])
print df.describe()

#analyze data by sex 0 unknown, 1 male,2 female
dSelect = df[["sex","snsupload_count"]] .groupby('sex').sum()#number of reposts in each sex
userSex = df.groupby('sex')#number of people in each sex
dSelect['number of people'] = userSex.size()
dSelect['average number of posts'] = dSelect['snsupload_count'].div(dSelect['number of people'], axis ='index' )
print dSelect

#analyze data by education
dSelect2 = df[["education_level","snsupload_count"]] .groupby('education_level').sum()#number of reposts in each sex
userEdu = df.groupby('education_level')#number of people in each sex
dSelect2['number of people'] = userEdu.size()
dSelect2['average number of posts'] = dSelect2['snsupload_count'].div(dSelect2['number of people'], axis ='index' )
print dSelect2

#analyze data by age
df = df.sort_values('age')
bins = np.arange(0,150,10)
age_ind = np.digitize(df['age'],bins)
dSelect3 = df[["age","snsupload_count"]].groupby(age_ind).sum()
userAge = df.groupby(age_ind)
dSelect3['number of people'] = userAge.size()
dSelect3['average number of posts'] = dSelect3['snsupload_count'].div(dSelect3['number of people'], axis ='index' )
dSelect3['age'] = dSelect3['age'].div(dSelect3['number of people'], axis ='index' )
dSelect3.rename(columns={'age':'average age'}, inplace = True)
print dSelect3.dtypes
dSelect3.index = dSelect3.index*10
print dSelect3

#analyze data by friend_count
df = df.sort_values('friend_count')
bins2 = np.around(np.logspace(0,4,num=10),decimals=2)
friend_ind = np.digitize(df['friend_count'],bins2)
dSelect4 = df[["friend_count","snsupload_count"]].groupby(friend_ind).sum()
userFriend = df.groupby(friend_ind)
dSelect4['number of people'] = userFriend.size()
dSelect4['average number of posts'] = dSelect4['snsupload_count'].div(dSelect4['number of people'], axis ='index' )
dSelect4['friend_count'] = dSelect4['friend_count'].div(dSelect4['number of people'], axis ='index' )
dSelect4.rename(columns={'friend_count':'average number of friends'}, inplace = True)
print dSelect4.dtypes
dSelect4.index = bins2.T[1:]
print dSelect4

dSelect.to_csv('user_sex_all.txt',index=True,header = True)
dSelect2.to_csv('user_edu_all.txt',index=True,header = True)
dSelect3.to_csv('user_age_all.txt',index=True,header = True)
dSelect4.to_csv('user_friend_all.txt',index=True,header = True)
