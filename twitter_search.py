import requests
import os
import json

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = bearer_token

num_results = 100
search_url = "https://api.twitter.com/2/tweets/search/recent/?max_results="+str(num_results)

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {'query': '@urbancompany_UC -is:reply -is:retweet -from:urbancompany_UC -from:UC_Assist', 'tweet.fields': 'author_id'}


def bearer_oauth(r):
	"""
	Method required by bearer token authentication.
	"""

	r.headers["Authorization"] = f"Bearer {bearer_token}"
	r.headers["User-Agent"] = "v2RecentSearchPython"
	return r

def connect_to_endpoint(url, params):
	response = requests.get(url, auth=bearer_oauth, params=params)
	print(response.status_code)
	if response.status_code != 200:
		raise Exception(response.status_code, response.text)
	return response.json()


def main():
	json_response = connect_to_endpoint(search_url, query_params)
	with open('tweets.txt', 'w') as f:
		json.dump(json_response, f)

if __name__ == "__main__":
	main()