import requests
import copy
import json

num_results = 100
search_url = "https://api.twitter.com/2/tweets/search/recent/?max_results="+str(num_results)
user_url = "https://api.twitter.com/2/users/"

query_params = {'query': '@urbancompany_UC -is:reply -is:retweet -from:urbancompany_UC -from:UC_Assist', 'tweet.fields': 'author_id'}

#Method required by bearer token authentication.
def bearer_oauth(r):
	r.headers["Authorization"] = f"Bearer {bearer_token}"
	r.headers["User-Agent"] = "v2RecentSearchPython"
	return r

def connect_to_endpoint(url, params, bearer_token):
	def bearer_oauth(r):
		r.headers["Authorization"] = f"Bearer {bearer_token}"
		r.headers["User-Agent"] = "v2RecentSearchPython"
		return r
	
	response = requests.get(url, auth=bearer_oauth, params=params)
	print(response.status_code)
	if response.status_code != 200:
		raise Exception(response.status_code, response.text)
	return response.json()

def get_tweets(bearer):
	out = connect_to_endpoint(search_url, query_params, bearer)
	with open('tweets.txt', 'w') as f:
		json.dump(out, f)
	return out

def filter_uc(tweets):
	out = []
	for i in tweets:
		bio = connect_to_endpoint(user_url+i['author'], {'user.fields': 'description'})['data']['description']
		if '@urbancompany_UC' not in bio:
			out.append(i)
	return out
