import pandas as pd
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import dateutil.relativedelta
pd.options.display.encoding = sys.stdout.encoding


fp = codecs.open("file", "r", encoding='utf-8')



df_reshare = pd.read_csv(fp,index_col=0,header = 0)
df_reshare.columns = {"fdate_cd","bizuin_md5","appmsgid","itemids","url","len",”posttime”,”uin_md5”,”reshare_timestamp”,”pre_published”,”cur_published”,”cur_chatmember_count”}
df_reshare = pd.DataFrame({'pre_published': df_reshare[‘pre_published’],'cur_published': df_reshare[‘cur_published’], ‘userId’: df_reshare[‘uin_md5’]});
df_reshare = df_reshare.drop_duplicates(keep = 'first')
df_reshare = df_reshare.drop_duplicates(subset= 'cur_published', keep = 'first')


preid = df_reshare['pre_published'].values.tolist();
curid =df_reshare['cur_published'].values.tolist();
userid =df_reshare['uin_md5'].values.tolist();






# convert userid to a node intergers
#useridList = [0]*userid.length
nonRepeatingUserid = []
for elem in userid:
	if elem is nonRepeatingUserid:
		elem = nonRepeatingUserid.index(elem)
	else:
		nonRepeatingUserid.append(elem)
elem = nonRepeatingUserid.index(elem)
print userid



for id in preid:
	if id not in curid:
		id = -1
	else:
		id = userid(curid.index(id))
print preid

output = pd.DataFrame({“from”: preid, “to”: userid})
newFrame.to_csv('output', sep=',')
		