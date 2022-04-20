import json
import requests
from tqdm import tqdm
from transformers import pipeline

with open('data.txt', 'r') as f:
	texts = f.read().split('\n')[:-1]

results = []
classifier = pipeline("sentiment-analysis")
for i in tqdm(range(100)):
	results.extend(classifier(texts[i*200:(i+1)*200]))

with open('results_sentiment.txt', 'w') as f:
	json.dump(results, f)
