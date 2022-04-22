from transformers import pipeline
import twitter_search as ts
import sentiment
import requests
import json
from tqdm import tqdm

with open('BEARER_TOKENS') as f:
	twitter_bearer, hf_bearer = f.read().split('\n')[:-1]

SENTENCE_SIMILARITY = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": hf_bearer}

classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

def query(payload):
	response = requests.post(SENTENCE_SIMILARITY, headers=headers, json=payload)
	return response.json()

def hf_query(q_params, sarcasm_q):
	similarity_score = query(q_params)
	sarcasm_score = query(sarcasm_q)
	try:
		_ = similarity_score[0]
		return similarity_score, sarcasm_score
	except:
		hf_query(q_params, sarcasm_q)

#data = ts.get_tweets(twitter_bearer)['data']
with open('dataset.json') as f:
	data = json.load(f)

def main(data):
	#pos, neg = sentiment.get_results(data, classifier)
	pos, neg = [], []
	for i in data:
		if i['label'] == 'POSITIVE':
			pos.append(i)
		else:
			neg.append(i)

	q_params = {'inputs': {
		'source_sentence': 'The service was excellent. The person was polite and professional',
		'sentences': [i['text'] for i in pos]
	}}

	sarcasm_q = {'inputs': {
		'source_sentence': "Bad customer care. Amazing!",
		'sentences': [i['text'] for i in pos]
	}}

	similarity_score, sarcasm_score = hf_query(q_params, sarcasm_q)

	out_pos, out_neg = [], []
	for i in range(len(pos)):
		if similarity_score[i] >= 0.3 and sarcasm_score[i] <= 0.45:
			out_pos.append(pos[i])
		else:
			out_neg.append(pos[i])
	out_neg.extend(neg)
	return out_pos, out_neg

#data = ts.get_tweets(twitter_bearer)['data']
#with open('dataset.json') as f:
#	data = json.load(f)
data = []
with open('positive_tweets.json') as f:
	data.extend(json.load(f))
with open('negative_tweets.json') as f:
	data.extend(json.load(f))

final_pos, final_neg = [], []
for i in tqdm(range(100)):
	temp_pos, temp_neg = main(data[i*370:(i+1)*370])
	final_pos.extend(temp_pos)
	final_neg.extend(temp_neg)

with open('positive_tweets_v2.json', 'w') as out:
	json.dump(final_pos, out)

with open('negative_tweets_v2.json', 'w') as out:
	json.dump(final_neg, out)

assert len(final_pos)+len(final_neg) == len(data)
