import pandas as pd
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import dateutil.relativedelta
pd.options.display.encoding = sys.stdout.encoding


fp = codecs.open("file.txt", "r", encoding='utf-8')



df_reshare = pd.read_csv(fp,index_col=0,header = 0)
df_reshare.columns = {"fdate_cd","bizuin_md5","appmsgid","itemids","url","len","posttime","uin_md5","reshare_timestamp","pre_published","cur_published","cur_chatmember_count"}
df_reshare = pd.DataFrame({'pre_published': df_reshare['pre_published'],'cur_published': df_reshare['cur_published'], 'userId': df_reshare['uin_md5']});
print "before removing dup ", df_reshare
df_reshare = df_reshare.drop_duplicates(keep = 'first')
print df_reshare
df_reshare = df_reshare.drop_duplicates(subset= 'cur_published', keep = 'first')


preid = df_reshare['pre_published'].values.tolist();
curid =df_reshare['cur_published'].values.tolist();
userid =df_reshare['userId'].values.tolist();
#print df_reshare






# convert userid to a node intergers
useridList = [0]*len(userid)
print userid
nonRepeatingUserid = []
i = 0
for elem in userid:
  if elem is nonRepeatingUserid:
    useridList[i] = nonRepeatingUserid.index(elem)
  else:
    nonRepeatingUserid.append(elem)
    useridList[i] = nonRepeatingUserid.index(elem)
  i += 1
print useridList



for id in preid:
  if id not in curid:
    id = -1
  else:
    id = useridList(curid.index(id))
print preid

output = pd.DataFrame({"from": preid, "to": userid})
output.to_csv('output', sep=',')