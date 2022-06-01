from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset
import tensorflow as tf
import twitter_search as ts
import pandas as pd
import numpy as np

with open('BEARER_TOKENS') as f:
    bearer = f.read().split('\n')[0]

model = TFAutoModelForSequenceClassification.from_pretrained('./fine-tuned')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

def main(data):
    sentences = [i['text'] for i in data]
    sentences = Dataset.from_pandas(pd.DataFrame({'text': sentences}))
    sentences = sentences.map(tokenize_function)

    tf_dataset = sentences.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=1
    )

    preds = model.predict(tf_dataset)['logits']
    preds = np.argmax(preds, axis=1)
    return preds

data = ts.get_tweets(bearer)['data']
outputs = main(data)
assert len(outputs) == len(data)

pos, neg = [], []
for i in range(len(outputs)):
    if outputs[i] == 0:
        neg.append(data[i]['text'])
    else:
        pos.append(data[i]['text'])

print("<" + '-'*10 + 'NON POSITIVE CUSTOMER REVIEW TWEETS' + '-'*10 + '>')
for i in neg:
    print(i)

print()
print("<" + '-'*10 + 'POSITIVE CUSTOMER REVIEW TWEETS' + '-'*10 + '>')
for i in pos:
    print(i)
