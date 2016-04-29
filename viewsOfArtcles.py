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
fView = codecs.open("/data/wechat_data/article_read","r", encoding='utf-8')
tops = codecs.open("/data/stanford/p_ruixilin/allDistribution","r", encoding='utf-8')
articles = pd.read_csv(tops, sep = ",", header = None,dtype='unicode',encoding="utf-8")
articles.columns = ["id","appmsgid", "bizuin_md5","itemids","count"]
data = pd.read_csv(fp, sep = "\t", header = None,dtype='unicode',encoding="utf-8")
data.columns = ["fdate_cd", "bizuin_md5", "appmsgid", "itemids", "title", "url", "len", "posttime", "uin_md5", "reshare_timestamp","pre_published", "cur_published", "cur_chatmember_count"];
dataView = pd.read_csv(fView, sep = "\t", header = None,dtype='unicode',encoding="utf-8")
dataView.columns = ["fdate_cd", "bizuin_md5", "appmsgid", "itemidx", "title", "url", "len", "posttime", "uin_md5", "read_timestamp", "read_scene","platform"];

listBizuin = []
listAppid = []
listItemid = []
listViews = []

for i in xrange(0,2000):
    row = articles.iloc[i]
    #print row
    bizuin =row['bizuin_md5']
    appmsgid = row['appmsgid']
    #print "bizuin",bizuin
    #print "appmsgid",appmsgid
    itemid = row['itemids']
    #print "itemid", itemid
    info = data[(data.bizuin_md5 == bizuin) & (data.appmsgid == appmsgid) & (data.itemids == itemid)]
    infoView = dataView[(dataView.bizuin_md5 == bizuin) & (dataView.appmsgid == appmsgid) & (dataView.itemidx == itemid)]
    result = pd.DataFrame({'posttime': info['posttime'],'resharetime': info['reshare_timestamp']}) 
    resultView = pd.DataFrame({'read_timestamp': infoView['read_timestamp']})
    #print(info)
    result = result.sort(columns = 'resharetime', ascending = True)
    resultView = resultView.sort(columns = 'read_timestamp', ascending = True)


    # find reshare time at first 10, 20, 40, 80 reshares
    #reshareSteps = [10]#,20,40,80]
    views = 0
    step = 10
    
    if len(result.resharetime) >= 10:
        #for step in reshareSteps:
        lastReshareAtSteps = result.iloc[step-1]['resharetime']
        for k in xrange(len(resultView)):
            if resultView.iloc[k]['read_timestamp'] <= lastReshareAtSteps:
                views += 1
            else:
                break
        #print "step is %s and view is %s  ", step, views
    listBizuin.append(bizuin)
    listAppid.append(appmsgid)
    listItemid.append(itemid)
    listViews.append(views)
    print "i is ", i

result1 = pd.DataFrame({'bizuin_md5':listBizuin,'appmsgid':listAppid, 'itemids': listItemid, 'views':listViews})
filename ="/data/stanford/p_ruixilin/viewsResult1.txt"
result1.to_csv(filename)   
#df = pd.DataFrame(some_list, columns=["colummn"])
#>>> df.to_csv('list.csv', index=False)     











