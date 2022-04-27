import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = tf.keras.models.load_model("model_v2.h5")

n = int(input("Number of sentences: "))
sentences = []
for i in range(n):
	sentences.append(input(f"Sentence {i+1}: "))

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = embedding_model(**encoded_input)

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
sentence_embeddings = np.array(sentence_embeddings)

preds = model.predict(sentence_embeddings)
assert len(preds) == n

for i in range(n):
	if preds[i] >= 0.5:
		print(f"POSITIVE: {sentences[i]}")
	else:
		print(f"NEGATIVE: {sentences[i]}")
