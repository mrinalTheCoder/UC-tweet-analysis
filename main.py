import twitter_search as ts
import sentiment
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json

with open('BEARER_TOKENS') as f:
	twitter_bearer, hf_bearer = f.read().split('\n')[:-1]

SENTENCE_SIMILARITY = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": hf_bearer}

def query(payload):
	response = requests.post(SENTENCE_SIMILARITY, headers=headers, json=payload)
	return response.json()

#data = ts.get_tweets(twitter_bearer)['data']
with open('tweets.txt') as f:
	data = json.load(f)['data']
pos, neg = sentiment.get_results(data)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosine_similarity(a, b):
	return torch.sum(a*b)/pow(torch.sum(torch.square(a))*torch.sum(torch.square(b)), 0.5)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sentences = ["The service was excellent. The person was polite and professional", "Very happy with UC's terrible customer care."]
sentences += [tweet['text'] for tweet in pos]

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sim_embeddings, sarc_embeddings = sentence_embeddings[0], sentence_embeddings[1]
sentence_embeddings = sentence_embeddings[2:]

similarity_score = [cosine_similarity(sim_embeddings, i) for i in sentence_embeddings]
sarcasm_score = [cosine_similarity(sarc_embeddings, i) for i in sentence_embeddings]

assert len(sarcasm_score) == len(pos)
assert len(similarity_score) == len(pos)

for i in range(len(pos)):
	if similarity_score[i] >= 0.3 and sarcasm_score[i] <= 0.45:
		print(pos[i])
