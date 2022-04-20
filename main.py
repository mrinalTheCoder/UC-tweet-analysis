import twitter_search as ts
import sentiment
import requests

SENTENCE_SIMILARITY = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": bearer}

def query(payload):
	response = requests.post(SENTENCE_SIMILARITY, headers=headers, json=payload)
	return response.json()

data = ts.get_tweets()['data']
pos, neg = sentiment.get_results(data)

q_params = {'inputs': {
	'source_sentence': 'The service was excellent. The person was polite and professional',
	'sentences': [i['text'] for i in pos]
}}
#sarcasm_q = {'inputs': {
#	'source_sentence': 'Bad customer care. Amazing!',
#	'sentences': [i['text'] for i in pos]
#}}

sarcasm_q = {'inputs': {
	'source_sentence': "Very happy with UC's terrible customer care.",
	'sentences': [i['text'] for i in pos]
}}

def hf_query():
	similarity_score = query(q_params)
	sarcasm_score = query(sarcasm_q)
	try:
		_ = similarity_score[0]
		return similarity_score, sarcasm_score
	except:
		hf_query()

similarity_score, sarcasm_score = hf_query()

for i in range(len(pos)):
	if similarity_score[i] >= 0.3 and sarcasm_score[i] <= 0.45:
		print(pos[i])
