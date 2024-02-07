from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
import numpy as np
import pickle
import re
from collections import defaultdict
from functools import partial
import argparse
from sentence_transformers import SentenceTransformer
from transformers import MLukeTokenizer, LukeModel

class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings).detach().numpy()


parser = argparse.ArgumentParser()
parser.add_argument('--ind', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default='sentence')
parser.add_argument('--suffix', type=str, default='')
args = parser.parse_args()
batch_size=100

pos = defaultdict(list)
neg = defaultdict(list)
if args.model=='roberta':
    feature_extractor = pipeline("feature-extraction",framework="pt",model="rinna/japanese-roberta-base", device=args.device)
elif args.model=='sentence':
    feature_extractor = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual', device=f'cuda:{args.device}')
elif args.model=='luke':
    feature_extractor = SentenceLukeJapanese("sonoisa/sentence-luke-japanese-base-lite")
with open(f'/home/yamanishi/project/airport/src/data/review/goodbad_all_{args.suffix}_{args.ind}.pkl', 'rb') as f:
    goodbad= pickle.load(f)
    
    sentences = []
    posnegs = []
    inds = []
    for i,(k,v) in enumerate(goodbad.items()):
        #print(i)
        matches = re.findall(r'「([^」]+)」', v)
        negative_ind = v.find('ネガティブな')
        posneg = [(v.find(m)<negative_ind) for m in matches]
        
        for m, pn in zip(matches, posneg):
            sentences.append(m)
            posnegs.append(pn)
            inds.append(k)
            #print(len(sentences))
            if len(sentences)==batch_size:
                if args.model=='roberta':
                    embs = feature_extractor(sentences,return_tensors = "pt")
                    embs = [emb.squeeze(0).numpy().mean(axis=0) for emb in embs]
                elif args.model=='sentence':
                    embs = feature_extractor.encode(sentences)
                elif args.model=='luke':
                    embs = feature_extractor.encode(sentences, batch_size=len(sentences))
                    
                print(embs.shape)
                #print([emb.shape for emb in embs])
                
                #mbs = feature_extractor(sentences,return_tensors = "pt")[0].numpy().mean(axis=0)
                for sent, pn, ind, emb in zip(sentences, posnegs, inds, embs):
                    if pn:pos[ind].append((sent, emb))
                    else:neg[ind].append((sent, emb))
                    
                sentences = []
                posnegs = []
                inds = []
        #if i==100:break
if args.model=='roberta':
    embs = feature_extractor(sentences,return_tensors = "pt")
    embs = [emb.squeeze(0).numpy().mean(axis=0) for emb in embs]
elif args.model=='sentence':
    embs = feature_extractor.encode(sentences)
elif args.model=='luke':
    embs = feature_extractor.encode(sentences, batch_size=len(sentences))
    
for sent, pn, ind, emb in zip(sentences, posnegs, inds, embs):
    if pn:pos[ind].append((sent, emb))
    else:neg[ind].append((sent, emb))
#print(pos)
#print(neg)
                    
with open(f'/home/yamanishi/project/airport/src/data/review/pos_all_{args.suffix}_{args.ind}.pkl', 'wb') as f:
    pickle.dump(pos, f)
with open(f'/home/yamanishi/project/airport/src/data/review/neg_all_{args.suffix}_{args.ind}.pkl', 'wb') as f:
    pickle.dump(neg, f)
