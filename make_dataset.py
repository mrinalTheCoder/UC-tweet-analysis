import os
import json
from tqdm import tqdm

files = os.listdir('./twitter-feed')
tweets = []

for path in tqdm(range(len(files))):
	with open('./twitter-feed/'+files[path]) as f:
		x = json.load(f)
		for i in range(len(x)):
			if 'attachments' not in x[i]:
				continue
			for j in range(len(x[i]['attachments'])):
				if 'pretext' not in x[i]['attachments'][j]:
					continue
				tweet_id = x[i]['attachments'][j]['pretext'].split('/')[-1][:-1]
				text = x[i]['attachments'][j]['text']
				text = text.replace('<https:\/\/twitter.com\/', '')
				text = text.replace('>', '')
				text = text.replace('\n', ' ')
				text = text.split()
				text = list(map(lambda inp:inp.split('|')[-1], text))
				text = ' '.join(text)
				tweets.append({'id': tweet_id, 'text': text})

with open('dataset.json', 'w') as f:
	json.dump(tweets, f)
