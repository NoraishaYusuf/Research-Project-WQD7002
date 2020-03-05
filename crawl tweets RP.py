
from twitterscraper import query_tweets
import datetime as dt
import pandas as pd

tweets = query_tweets('@MyRapidKL', limit=None) #for rapidkl1.csv from 21/03/2006 to 15/09/2019
#tweets = query_tweets('RapidKL', limit=None) for rapidkl.csv
# crawled from 2006 to 19/08/2019)
tweetslist = (t.__dict__ for t in tweets)
df = pd.DataFrame(tweetslist)
df.to_csv('rapidkl1.csv',index = False)
#df2 =  df.copy()
#df2['test']=df2['timestamp']
#cols = list(df2.columns)
#cols = cols[:-6] + [cols[-1]] + cols[-6:-1]
#df3 = df2[cols]

df4 = pd.read_csv('rapidkl1.csv',index_col = False)
df4.dtypes

df4['date'] = pd.to_datetime(df4['timestamp']) 
df4 = df4[(df4['date'] > '2018-1-1') & (df4['date'] <= '2018-12-31')]
df4.to_csv('rapidkl_1_2018.csv',index = False)

df4['date'] = pd.to_datetime(df4['timestamp']) 
df4 = df4[(df4['date'] > '2019-1-1') & (df4['date'] <= '2019-6-30')]
df4.to_csv('rapidkl_1_2019.csv',index = False)

