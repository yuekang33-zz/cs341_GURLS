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
fp = codecs.open("results/18.txt", "r", encoding='utf-8')
art = pd.read_csv(fp, sep = ",", header = 0,encoding="utf-8")
art.columns = ("index","posttime", "resharecount");

numRow = art.shape[0]
postTime = art.iloc[0]['posttime']
lastReshareTime =art.iloc[0]['resharecount']
timeStep = 3600
timeSlots = (lastReshareTime - postTime)/timeStep+1

	#postTime = datetime.datetime.fromtimestamp(postTime)
	#lastReshareTime = datetime.datetime.fromtimestamp(lastReshareTime)
bins = np.zeros(timeSlots)
for i in xrange(timeSlots):
	bins[i] = postTime+i*timeStep

x = []
for j in xrange(timeSlots-1):
	#if j==0: 
	#	x.append("0"+"," + str(datetime.datetime.fromtimestamp(bins[j])))
	#else: 
		x.append(""+str(datetime.datetime.fromtimestamp(bins[j])))
print(x)
xAxis = pd.DataFrame({'x': x})
print xAxis
Xuniques= np.unique(x, return_inverse=True)
art['resharecount'] = pd.to_numeric(art['resharecount'], errors='coerce')
art['time'] = pd.cut(art['resharecount'],bins)
result = art[['time','resharecount']].groupby('time').count()
result.set_index(xAxis['x'])
#ax = plt.subplots()
ax =result.plot(kind='line', logy=False)
ax.set_xticklabels(list(xAxis['x']), rotation =45)

bx = result.plot(kind='bar', logy=False)
#plt.xticks(np.arange(min(list(xAxis['x'])), max(list(xAxis['x']))+1, 10.0))
x = []
for i in range(len(list(xAxis['x']))):
	x.append(list(xAxis['x'])[i])
	i = i + 100

bx.set_xticklabels(x, rotation =45)
#fig.set(xTicks = len(x), xTicksLabels = x)
plt.show()	
