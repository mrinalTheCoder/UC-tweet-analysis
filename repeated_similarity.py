import json
from tqdm import tqdm
import random
import sentiment
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline

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
def hf_query(ref_sentences, target_sentences, tokenizer, model):
    if 'embedding' not in ref_sentences[0]:
        encoded_input = tokenizer(
            [i['text'] for i in ref_sentences],
            padding=True, truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            model_output = model(**encoded_input)

        ref_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        for i in range(len(ref_sentences)):
            ref_sentences[i]['embedding'] = ref_embeddings[i]

    if 'embedding' not in target_sentences[0]:
        for batch in tqdm(range(len(target_sentences)//200 + 1)):
            target_slice = target_sentences[batch*200:(batch+1)*200]
            if len(target_slice) == 0:
                continue
            encoded_input = tokenizer([
                i['text'] for i in target_slice],
                padding=True, truncation=True, return_tensors='pt'
            )
            with torch.no_grad():
                model_output = model(**encoded_input)

            target_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            for i in range(len(target_slice)):
                target_slice[i]['embedding'] = target_embeddings[i]
            target_sentences[batch*200:(batch+1)*200] = target_slice

    ref_embeddings = [i['embedding'] for i in ref_sentences]
    target_embeddings = [i['embedding'] for i in target_sentences]

    sim_scores = [
        [cosine_similarity(ref_embeddings[j], target_embeddings[i]) for j in range(len(ref_sentences))]
    for i in range(len(target_sentences))]
    return np.array(sim_scores)

def iterate(refs, pos, tokenizer, model, NUM_REFS_PER_ITER, FINAL_THRESH):
    sim_scores = hf_query(refs, pos, tokenizer, model)
    sim_scores = np.average(sim_scores, axis=1)
    for i in range(len(pos)):
        current_score, new_score = pos[i]['score'], sim_scores[i]
        pos[i]['changed'] = (current_score >= FINAL_THRESH) != (new_score >= FINAL_THRESH)
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

def get_results(pos, NUM_ITERS=20):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # NUM_ITERS * NUM_REFS_PER_ITER = TOTAL_REF_PROPORTION of dataset
    MIN_ITERS = 2
    TOTAL_REF_PROPORTION = 0.05
    NUM_REFS_PER_ITER = int(TOTAL_REF_PROPORTION * len(pos)//NUM_ITERS)
    FINAL_THRESH = 0.395

    refs = [
        {'text': "The service was excellent. The person was polite and professional"},
        {'text': "Impressed with safe and hygienic service. Every tool was sanitised."}
    ]

    sim_scores = hf_query(refs, pos, tokenizer, model)
    sim_scores = np.average(sim_scores, axis=1)
    for i in range(len(pos)):
        pos[i]['score'] = sim_scores[i]

    pos = sorted(pos, reverse=True, key= lambda x:x['score'])

    all_refs = []
    for i in range(NUM_ITERS):
        refs, pos = iterate(refs, pos, tokenizer, model, NUM_REFS_PER_ITER, FINAL_THRESH)
        all_refs.extend(refs)
        changed_prop = sum([i['changed'] for i in pos])/len(pos)
        print(f"Iteration {i+1}: change proportion is {changed_prop}")

        if i < MIN_ITERS:
            continue
        if changed_prop < 0.01:
            print(f"Breaking after {i+1} iterations")
            break

    thresh_idx = bin_search([i['score'] for i in pos], FINAL_THRESH, 0, len(pos)-1)

    final_pos, final_neg = pos[:thresh_idx], pos[thresh_idx:]
    for i in all_refs+final_pos+final_neg:
        del i['embedding']
        del i['changed']
        del i['score']
    return all_refs, final_pos, final_neg

    # classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
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

if __name__ == '__main__':
    with open('sentiment_pos_tweets.json') as f:
        data = json.load(f)
    for i in data:
        i['changed'] = False
        del i['label']
        del i['score']
    all_refs, final_pos, final_neg = get_results(data)

    print("<---------- REFERENCE SENTENCES ---------->")
    for i in all_refs:
        del i['embedding']
        print(i)

    print()
    print("<---------- POSITIVE SENTENCES ---------->")
    for i in final_pos:
        del i['embedding']
        print(i)

    print()
    print("<---------- NEGATIVE SENTENCES ---------->")
    for i in final_neg:
        del i['embedding']
        print(i)
