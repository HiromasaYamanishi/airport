import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import MLukeTokenizer, LukeModel
from models.language_models import SentenceLukeJapanese
from sklearn.metrics.pairwise import cosine_similarity
import os
from functools import partial
from collections import defaultdict
import pickle

def find_connected_groups(matrix):
    def dfs(node, group):
        visited[node] = True
        group.append(node)
        for neighbour, isConnected in enumerate(matrix[node]):
            if isConnected and not visited[neighbour]:
                dfs(neighbour, group)

    n = len(matrix)
    visited = [False] * n
    groups = []

    for node in range(n):
        if not visited[node]:
            group = []
            dfs(node, group)
            groups.append(group)

    return groups

def groupby_sim(features, thresh=0.5):
    cos_sim = cosine_similarity(features)
    graph = cos_sim>=thresh
    group = find_connected_groups(graph)
    return group

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='')
    parser.add_argument('--thresh', type=float, default=0.7)
    args = parser.parse_args()
    feature_extractor = SentenceLukeJapanese("sonoisa/sentence-luke-japanese-base-lite")
    
    original_file_name = args.df_path.split('/')[-1].split('.')[0]+'_merged.csv'
    directory_name = '/'.join(args.df_path.split('/')[:-1])
    save_path = os.path.join(directory_name, original_file_name)
    #print(save_path)
    df = pd.read_csv(args.df_path, names=['index', 'spot', 'posneg', 'cluster_label', 'topics'])
    df_merged = []
    groups = defaultdict(partial(defaultdict, list))
    for spot in df['spot'].unique():
        df_tmp_pos = df[(df['spot']==spot)&(df['posneg']=='pos')].reset_index()
        df_tmp_neg = df[(df['spot']==spot)&(df['posneg']=='neg')].reset_index()
        if len(df_tmp_pos):
            pos_topic_embs = feature_extractor.encode(df_tmp_pos['topics'].values, batch_size=len(df_tmp_pos['topics'].values))
            pos_topics_sims = cosine_similarity(pos_topic_embs)
            pos_groups = groupby_sim(pos_topic_embs, thresh=args.thresh)
            pos_groups_rep = [p[0] for p in pos_groups] #一番最初のトピックを代表的なトピックとする
            df_merged.append(df_tmp_pos.loc[pos_groups_rep])
            #df_pos_rep = df_tmp_pos.loc[pos_groups_rep].reset_index()
            #df_pos_rep['groups'] = [' '.join([str(p_) for p_ in p]) for p in pos_groups]           
            #df_merged.append(df_pos_rep)
            groups[spot]['pos'] = pos_groups

        if len(df_tmp_neg):
            neg_topic_embs = feature_extractor.encode(df_tmp_neg['topics'].values, batch_size=len(df_tmp_neg['topics'].values))
            neg_topics_sims = cosine_similarity(neg_topic_embs)
            neg_groups = groupby_sim(neg_topic_embs, thresh=0.8)
            #print('neg groups', neg_groups)
            neg_groups_rep = [n[0] for n in neg_groups]
            df_merged.append(df_tmp_neg.loc[neg_groups_rep])
            groups[spot]['neg'] = neg_groups
            
    df_merged = pd.concat(df_merged)
    df_merged.to_csv(save_path, )
    save_name = args.df_path.split('/')[-1].split('.')[0]
    with open(f'../data/groups/group_{save_name}.pkl', 'wb') as f:
        pickle.dump(groups, f)
    # print(df_merged)
    # print(len(df_merged))
            # for ng in neg_groups:
            #     df_merged.append(df_tmp_pos.loc[ng[0]])
            #     if len(ng)>=2:
            #         print(df_tmp_neg.loc[ng])
        #print('pos emb sim', pos_topics_sims)
        #print('neg emb sim', neg_topics_sims)