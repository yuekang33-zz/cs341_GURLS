import pandas as pd
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import dateutil.relativedelta
pd.options.display.encoding = sys.stdout.encoding

#for k in xrange(19):
#	fp = codecs.open("/Users/ruixilin/downloads/results/"+str(k)+".txt", "r", encoding='utf-8')
'''
fp = codecs.open("results/0.txt", "r", encoding='utf-8')
art = pd.read_csv(fp, sep = ",", header = 0,encoding="utf-8")
art.columns = ("index","posttime", "resharetime");
		#print art

result2 = pd.DataFrame({'count':art.groupby('resharetime').size()})
print(result2)
result2.plot(kind='bar',logy=False)
plt.show()

'''
fp = codecs.open("file", "r", encoding='utf-8')

'''
art = pd.read_csv(fp, sep = ",", header = 0,encoding="utf-8")
art.columns = ("index","fdate_cd", "bizuin_md5", "appmsgid","itemids", "url", "len", "postime","uin_md5", "reshare_timestamp", "pre_published", "cur_published", "cur_chatmember_count");
print art.shape


pre = pd.DataFrame(art["pre_published"])
cur = pd.DataFrame(art["cur_published"])
resultB = pre.isin(cur)
print(art[resultB['pre_published']].index.tolist())
print(resultB) 
'''

df_reshare = pd.read_csv(fp,index_col=0,header = 0)
df_reshare = df_reshare.drop_duplicates(keep = 'first')
df_reshare = df_reshare.drop_duplicates(subset= 'cur_published', keep = 'first')
preid = pd.DataFrame(df_reshare[['pre_published']])
curid = pd.DataFrame(df_reshare[['cur_published']])
#print preid['pre_published']
idintersect = pd.DataFrame()
idintersect['idintersect'] = preid['pre_published'].isin(curid['cur_published'])
#&curid['cur_published'].isin(preid['pre_published'])
#idintersect.columns.values[1] ="idintersect"
bothid = pd.concat([preid,curid,idintersect],axis=1)
preid_source = bothid[bothid['idintersect']==False]['pre_published']
preid_source = preid_source.drop_duplicates(keep = 'first')
preid_source = pd.DataFrame(preid_source)
preid_source.columns = ["pre_published"]
print preid_source
bothid = bothid[bothid['idintersect']==False]
fakeSource = []
for i in range(preid_source.shape[0]):
	fakeSource.append("0")

fakeFrame = pd.DataFrame({'pre': fakeSource,'cur': preid_source['pre_published']});
orignialFrame = pd.DataFrame({'pre': df_reshare['pre_published'],'cur': df_reshare['cur_published']})
newFrame = orignialFrame.append(fakeFrame)
print newFrame.shape
newFrame.to_csv('fakeFile', sep=',')

