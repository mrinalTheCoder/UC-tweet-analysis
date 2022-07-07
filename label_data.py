from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sentiment
import repeated_similarity
import requests
import json
from tqdm import tqdm

classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


out =[{}]
def main(data):
    pos, neg = [], []
    for i in tqdm(range(100)):
        out = sentiment.get_results(data[i*(len(data)//100):(i+1)*(len(data)//100)], classifier)
        pos.extend(out[0])
        #positive tweets using bert-modal
        neg.extend(out[1])
        #negative tweets using bert-modal

    # use for Debugging for runnung data on bert model to get  postive and negative by commenting above code
    #-------------------------
    # pos, neg = [], []
    # for i in data:
        # if i['label'] == 'POSITIVE':
            # pos.append(i)
        # else:
            # neg.append(i)
    #--------------------------

    sim_refs, sim_pos, sim_neg = repeated_similarity.get_results(pos)
    #calculate final positive and negative sentence
    return sim_refs+sim_pos, neg+sim_neg

with open('dataset.json') as f:
    data = json.load(f)    

# use for Debugging for runnung initial dataset of bert model to get first output 
#-------------------------
# data = []
# with open('sentiment_pos_tweets.json') as f:
    # data.extend(json.load(f))
# with open('sentiment_neg_tweets.json') as f:
    # data.extend(json.load(f))
#----------------------------

final_pos, final_neg = main(data)

with open('positive_tweets.json', 'w') as out:
    json.dump(final_pos, out)

with open('negative_tweets.json', 'w') as out:
    json.dump(final_neg, out)

assert len(final_pos)+len(final_neg) == len(data)
