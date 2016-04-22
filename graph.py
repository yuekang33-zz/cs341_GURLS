import pandas as pd
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import dateutil.relativedelta
from snap import *
pd.options.display.encoding = sys.stdout.encoding


fp = codecs.open("file", "r", encoding='utf-8')



df_reshare = pd.read_csv(fp,index_col=0,sep = ",",header = None)
#print df_reshare
df_reshare.columns = ["fdate_cd","bizuin_md5","appmsgid","itemids","url","len","posttime","uin_md5","reshare_timestamp","pre_published","cur_published","cur_chatmember_count"]
#df_reshare = pd.DataFrame()
#print "before removing dup ", df_reshare['appmsgid']
df_reshare = pd.DataFrame({'pre_published': df_reshare['pre_published'],'cur_published': df_reshare['cur_published'], 'userId': df_reshare['uin_md5']});

df_reshare = df_reshare.drop_duplicates(keep = 'first')
#print df_reshare
df_reshare = df_reshare.drop_duplicates(subset= 'cur_published', keep = 'first')
#print "len after removing dup",len(df_reshare)


preid = df_reshare['pre_published'].values.tolist();
curid =df_reshare['cur_published'].values.tolist();
userid =df_reshare['userId'].values.tolist();
#print len(userid)






# convert userid to a node intergers
useridList = [0]*len(userid)
#print userid
nonRepeatingUserid = []
i = 0
for elem in userid:
  if elem in nonRepeatingUserid:
    useridList[i] = nonRepeatingUserid.index(elem)+1
  else:
    nonRepeatingUserid.append(elem)
    useridList[i] = nonRepeatingUserid.index(elem)+1
  i += 1
print len(useridList)



for i in xrange(len(preid)):
  if preid[i] not in curid:
    preid[i] = 0
  else:
    preid[i] = useridList[curid.index(preid[i])]
print preid

output = pd.DataFrame({"from": preid, "to": userid})
output.to_csv('output', sep=',')




#construct graph
G1 = TNGraph.New()
for nodeVal in useridList:
  G1.AddNode(0)
  G1.AddNode(nodeVal)
for i in xrange(len(preid)):
  if preid[i] != -1:
      G1.AddEdge(preid[i], useridList[i])











