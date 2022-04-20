import json
from tqdm import tqdm
import requests

with open('data.txt', 'r') as f:
	data = f.read().split('\n')[:-1]

with open('results_sentiment.txt', 'r') as f:
	s_res = json.load(f)

with open('sim_data.txt', 'r') as f:
	sim_res = f.read().split('\n')[:-1]
	sim_res = list(map(float, sim_res))

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": bearer}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

sarc_res = []
for i in tqdm(range(100)):
	sarc_res.extend(query({'inputs': {
		'source_sentence': "Very happy with UC's terrible customer care.",
		'sentences': data[i*200:(i+1)*200]
	}}))

pos, neg = [], []
for i in range(len(data)):
	if s_res[i]['label'] == 'POSITIVE' and sim_res[i] >= 0.3 and sarc_res[i] <= 0.45:
		pos.append(data[i])
	else:
		neg.append(data[i])

print("-------------- POSITIVE --------------")
for i in pos:
	print(i)

print()
print("-------------- NEGATIVE--------------")
for i in neg:
	print(i)

print(f"{len(pos)} positive feedback tweets")
print(f"{len(neg)} negative feedback tweets")
