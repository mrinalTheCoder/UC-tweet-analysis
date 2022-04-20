import json
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": bearer}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

with open('data.txt', 'r') as f:
	data = f.read().split('\n')[:-1]

with open('results_sentiment.txt', 'r') as f:
	s_data = json.load(f)

r_data = []
for i in range(100):
	r_data.extend(query({'inputs': {
		'source_sentence': 'The service was excellent. The person was polite and professional.',
		'sentences': data[i*200:(i+1)*200]
	}}))

	print(f'{i+1}% data fetched')

print('r_data fetched')

with open('sim_data.txt', 'w') as f:
	for i in r_data:
		f.write(str(i) + '\n')
