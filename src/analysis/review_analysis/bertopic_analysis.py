from bertopic import BERTopic
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

cluster_model = KMeans(n_clusters=20) 
sentence_model = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")
topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=cluster_model, calculate_probabilities=True, verbose=True)
df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/kagawa_review.csv')
docs = list(df_kagawa['review'].values)
topics, probs = topic_model.fit_transform(docs)

with open('../data/topic/topic_model.pkl', 'wb') as f:
    pickle.dump(topic_model, f)
    
d_topic = {'topics': topics, 'probs': probs}
with open('../data/topic/topics.pkl', 'wb') as f:
    pickle.dump(d_topic, f)
    
    
