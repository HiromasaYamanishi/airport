import pandas as pd
df = pd.read_csv('/home/yamanishi/project/airport/src/data/kagawa_review.csv')
#df = df[df['pref']=='香川県']

import spacy
nlp = spacy.load('ja_ginza_electra')
for i in range(100):
    doc = nlp(df['review'].values[i])
    for sent in doc.sents:
        for token in sent:
            print(
                token.i,
                token.orth_,
                token.lemma_,
                token.norm_,
                token.morph.get("Reading"),
                token.pos_,
                token.morph.get("Inflection"),
                token.tag_,
                token.dep_,
                token.head.i,
            )
        print('EOS')