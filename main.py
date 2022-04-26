from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sentiment
import requests
import json
from tqdm import tqdm

classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosine_similarity(a, b):
	return torch.sum(a*b)/pow(torch.sum(torch.square(a))*torch.sum(torch.square(b)), 0.5)

def hf_query(ref_sentences, target_sentences):
	sentences = ref_sentences + target_sentences
	encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
	with torch.no_grad():
		model_output = model(**encoded_input)
	
	sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
	ref_embeddings = sentence_embeddings[:len(ref_sentences)]
	target_embeddings = sentence_embeddings[len(ref_sentences):]

	sim_scores = [
		[cosine_similarity(ref_embeddings[j], target_embeddings[i]) for j in range(len(ref_sentences))]
	for i in range(len(target_sentences))]
	return sim_scores

def main(data):
	#pos, neg = sentiment.get_results(data, classifier)
	pos, neg = [], []
	for i in data:
		if i['label'] == 'POSITIVE':
			pos.append(i)
		else:
			neg.append(i)

	similarities = hf_query([
		"The service was excellent. The person was polite and professional",
		"Impressed with safe and hygienic service. Every tool was sanitised."
	], [i['text'] for i in pos])
	
	out_pos, out_neg = [], []
	for i in range(len(pos)):
		#if similarity_score[i] >= 0.3 and sarcasm_score[i] <= 0.45:
		for j in range(len(similarities[0])):
			if similarities[i][j] >= 0.3:
				out_pos.append(pos[i])
				break
		else:
			out_neg.append(pos[i])
	out_neg.extend(neg)
	return out_pos, out_neg

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

with open('positive_tweets.json', 'w') as out:
	json.dump(final_pos, out)

with open('negative_tweets.json', 'w') as out:
	json.dump(final_neg, out)

assert len(final_pos)+len(final_neg) == len(data)
