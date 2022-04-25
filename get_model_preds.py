import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_preds(data):
	tweets = [i['text'] for i in data]
	tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
	transformer_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

	encoded_input = tokenizer(tweets, padding=True, truncation=True, return_tensors='pt')
	with torch.no_grad():
		model_output = transformer_model(**encoded_input)
	
	sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
	sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

	model = tf.keras.models.load_model('model.h5')
	return model.predict(np.array(sentence_embeddings))
