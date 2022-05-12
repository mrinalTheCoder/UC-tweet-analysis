import json
import random
from tqdm import tqdm
import sentiment
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

with open('positive_tweets.json') as f:
    pos_data = json.load(f)

pos = pos_data[2000:4000]
for i in range(len(pos_data)):
    pos_data[i]['id'] = i
    pos_data[i]['score'] = 0
    del pos_data[i]['label']

# NUM_ITERS * NUM_REFS_PER_ITER = TOTAL_REF_PROPORTION of dataset
TOTAL_REF_PROPORTION = 0.05
NUM_ITERS = 5
NUM_REFS_PER_ITER = int(TOTAL_REF_PROPORTION * len(pos)//NUM_ITERS)
FINAL_THRESH = 0.395

def bin_search(arr, x, low, high):
    while low < high:
        mid = (low+high)//2
        if arr[mid] == x:
            return mid
        if arr[mid] > x:
            low = mid+1
        else:
            high = mid-1
    return low

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosine_similarity(a, b):
    temp = torch.sum(a*b)/pow(torch.sum(torch.square(a))*torch.sum(torch.square(b)), 0.5)
    return np.array(temp)

# returns (num_ref_sentences, num_target_sentences) 2d array of sim scores
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
    return np.array(sim_scores)

def iterate(refs, pos):
    sim_scores = hf_query([i['text'] for i in refs], [i['text'] for i in pos])
    sim_scores = np.average(sim_scores, axis=1)
    for i in range(len(pos)):
        pos[i]['score'] = sim_scores[i]
    pos = sorted(pos, reverse=True, key = lambda x:x['score'])
    
    new_ref_idxs = random.sample(list(range(NUM_REFS_PER_ITER*50)), NUM_REFS_PER_ITER)
    new_refs, new_pos = [], []
    for i in range(len(pos)):
        if i in new_ref_idxs:
            new_refs.append(pos[i])
        else:
            new_pos.append(pos[i])

    return new_refs, new_pos

def get_results(pos):
    refs = [
        {'text': "The service was excellent. The person was polite and professional"},
        {'text': "Impressed with safe and hygienic service. Every tool was sanitised."}
    ]

    sim_scores = np.array(hf_query([i['text'] for i in refs], [i['text'] for i in pos]))
    sim_scores = np.average(sim_scores, axis=1)
    for i in range(len(pos)):
        pos[i]['score'] = sim_scores[i]

    pos = sorted(pos, reverse=True, key= lambda x:x['score'])

    all_refs = []
    for i in tqdm(range(NUM_ITERS)):
        refs, pos = iterate(refs, pos)
        all_refs.extend(refs)

    thresh_idx = bin_search([i['score'] for i in pos], FINAL_THRESH, 0, len(pos)-1)

    final_pos, final_neg = pos[:thresh_idx], pos[thresh_idx:]
    return final_pos, final_neg

    # orig_scores = [i['score'] for i in pos[thresh_idx-10:thresh_idx+10]]
    # temp, _ = sentiment.get_results(pos[thresh_idx-10:thresh_idx+10], classifier)
    # for i in range(len(temp)):
        # temp[i]['orig_score'] = orig_scores[i]
        # del temp[i]['label']

    # temp = sorted(temp, reverse=True, key=lambda x:x['score'])
    # for i in temp:
        # i['score'] = i['orig_score']
        # del i['orig_score']
    # pos[thresh_idx-10:thresh_idx+10] = temp

"""
print("<---------- REFERENCE SENTENCES ---------->")
for i in all_refs:
    print(i)

print()
print("<---------- POSITIVE SENTENCES ---------->")
for i in final_pos:
    print(i)

print()
print("<---------- NEGATIVE SENTENCES ---------->")
for i in final_neg:
    print(i)
"""
