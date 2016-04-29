# zongyizhang/anaconda2/bin/python parseToframe.py
#-*- encoding:utf-8 -*-
import pandas as pd
import sys
import os
import codecs
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.encoding = sys.stdout.encoding

fp = codecs.open("/data/wechat_data/article_reshare","r", encoding='utf-8')
tops = codecs.open("/data/stanford/p_yuekang/allDistribution","r", encoding='utf-8')
articles = pd.read_csv(tops, sep = ",", header = None,dtype='unicode',encoding="utf-8")
articles.columns = ("id","appmsgid", "bizuin_md5","itemids","count")
data = pd.read_csv(fp, sep = "\t", header = None,dtype='unicode',encoding="utf-8")
data.columns = ["fdate_cd", "bizuin_md5", "appmsgid", "itemids", "title", "url", "len", "posttime", "uin_md5", "reshare_timestamp","pre_published", "cur_published", "cur_chatmember_count"];
for i in range(0,19):
	row = articles.iloc[i]
	print row
	#search = pd.DataFrame({'bizuin':row['bizuin_md5'], 'appmsgid':row['appmsgid'], 'itemid':row['itemids']})

	bizuin =row['bizuin_md5']
	appmsgid = row['appmsgid']
	print(bizuin)
	itemid = row['itemids']
	info = data[(data.bizuin_md5 == bizuin) & (data.appmsgid == appmsgid) & (data.itemids == itemid)]
	#print(i)
#info = data.loc['bizuin'==bizuin]
#	data['bizuin_md5'] = str(data['bizuin_md5'])
	#data['appmsgid']= data['appmsgid'].to_string()
	#data['itemids']= data['itemids'].to_string()
	#print(data['appmsgid'])
#	info = data.loc[data['bizuin_md5']==bizuin & data['appmsgid']==appmsgid]
	#print(info)
#info = data.loc[str(data['bizuin_md5'])==bizuin]
	#info = data.loc[(str(data['bizuin_md5'])==bizuin) && (str( data['appmsgid'])==appmsgid) &&(str(data['itemids']) == itemid)]
	result = pd.DataFrame({'posttime': info['posttime'],'resharetime': info['reshare_timestamp']}) 
	#print(info)
	result = result.sort(columns = 'resharetime', ascending = False)
	filename ="/data/stanford/p_ruixilin/articles/" + str(i) + ".txt"
	#filename = "articleData/"+str(i)+".txt"
	result.to_csv(filename)
