import twitter_search as ts
import sentiment
import json

data = ts.get_tweets()['data']
#with open('tweets.txt', 'r') as f:
#	data = json.load(f)['data']
pos, neg = sentiment.get_results(data)
pos = ts.filter_uc(pos)
for i in pos:
	print(i)
