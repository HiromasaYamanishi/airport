from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
import os
from transformers import default_data_collator
import random
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import Trainer, EarlyStoppingCallback
from transformers import DataCollatorForLanguageModeling
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import gensim
import argparse


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def preprocess_function(data):
    texts = [q.strip() for q in data["text"]]
    inputs = tokenizer(
        texts,
        max_length=450,
        truncation=True,
        padding='max_length', 
    )

    inputs['labels'] = torch.tensor(data['label'])

    return inputs

df = pd.read_csv('/home/yamanishi/project/airport/src/data/review_all_period_.csv')
d_sex = {'男性': 0, '女性': 1}
d_age = {'10代': 0, '20代': 0, '30代': 1, '40代': 1, '50代': 1, '60代': 2, '70代': 2,
         '80代': 2}
d_tag = {'カップル・夫婦': 0, '家族': 1, '友達同士': 2, '一人': 3}
ds = {'sex': d_sex, 'age': d_age, 'tag': d_tag}
parser = argparse.ArgumentParser()
parser.add_argument('--target', default='sex', type=str)
parser.add_argument('--load_data', action='store_true')
args = parser.parse_args()
le = LabelEncoder()
df = df.dropna(subset=args.target)
df = df[df[args.target].isin(list(ds[args.target].keys()))]
print(df[args.target].value_counts())
if args.target=='sex':
    df['label'] = df['sex'].map(ds[args.target]).astype(int)
    num_labels=2
elif args.target=='age':
    df['label'] = df['age'].map(d_age).astype(int)
    num_labels=3
elif args.target=='tag':
    df['label'] = df['tag'].map(d_tag).astype(int)
    num_labels=4
    

df['text'] = df['review']
df = df.dropna(subset='label')
df = df.dropna(subset='text')


set_seed(42)
data = df[['text', 'label']]

train, valid = train_test_split(data, test_size=0.3)
valid, test = train_test_split(valid, test_size=0.5)

ds_train = Dataset.from_pandas(train)
ds_valid = Dataset.from_pandas(valid)
ds_test = Dataset.from_pandas(test)


word2vec = gensim.models.Word2Vec.load('/home/yamanishi/project/trip_recommend/data/ja/ja.bin')
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")
new_vocab = [k for k in word2vec.wv.vocab.keys() if len(k)>2]
tokenizer.add_tokens(new_vocab, special_tokens=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained("rinna/japanese-roberta-base", num_labels=2).to(device)
model.resize_token_embeddings(len(tokenizer))

dataset = DatasetDict({
    "train": ds_train,
    "validation": ds_valid,
    "test": ds_test
})


if not args.load_data:
    tokenized_data = dataset.map(preprocess_function, batched=True)

    with open(f'/home/yamanishi/project/airport/src/data/review/tokenized_data_{args.target}.pkl', 'wb') as f:
        pickle.dump(tokenized_data , f)
    
else:
    with open(f'/home/yamanishi/project/airport/src/data/review/tokenized_data{args.target}.pkl', 'rb') as f:
        tokenized_data = pickle.load(f)

data_collator = default_data_collator
training_args = TrainingArguments(
    output_dir=f"./review_{args.target}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

data_collator = default_data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

    
trainer.train()