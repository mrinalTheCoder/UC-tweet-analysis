import json
from transformers import pipeline

with open('tweets.txt', 'r') as f:
	data = json.load(f)['data']

# data is list of objects, with id, author_id, text attributes

for i in range(len(data)):
	data[i]['text'] = data[i]['text'].replace('\n', ' ')
	l = data[i]['text'].split()
	new_l = []
	for j in l:
		if j[0] not in '@#':
			new_l.append(j)
		elif j == '@urbancompany_UC':
			new_l.append('UC')
	del l
	data[i]['text'] = ' '.join(new_l)

classifier = pipeline("sentiment-analysis")
result = classifier([data[i]['text'] for i in range(len(data))])

pos, neg = 0, 0
pos_results = []
for i in range(len(data)):
	temp = result[i]
	temp['text'] = data[i]['text']
	temp['id'] = data[i]['id']
	if temp['label'] == 'POSITIVE':
		pos_results.append(temp)
	pos += int(temp['label'] == 'POSITIVE')
	neg += int(not temp['label'] == 'POSITIVE')

# sort in descending order of positivity score
pos_results = sorted(pos_results, reverse=True, key=lambda x:x['score'])
for i in pos_results:
	print(i)
print()
print(f"{pos} positive tweets and {neg} negative tweets")
