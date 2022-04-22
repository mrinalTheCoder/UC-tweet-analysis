import json

# data is list of objects, with id, author_id, text attributes

def clean_data(data):
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
	return data

def get_results(data, classifier):
	data = clean_data(data)
	result = classifier([data[i]['text'] for i in range(len(data))])

	pos_results, neg_results = [], []
	for i in range(len(data)):
		temp = result[i]
		temp['text'] = data[i]['text']
		temp['id'] = data[i]['id']
		#temp['author'] = data[i]['author_id']
		if temp['label'] == 'POSITIVE':
			pos_results.append(temp)
		else:
			neg_results.append(temp)
	#pos_results = sorted(pos_results, reverse=True, key=lambda x:x['score'])
	#neg_results = sorted(neg_results, reverse=True, key=lambda x:x['score'])
	return pos_results, neg_results
