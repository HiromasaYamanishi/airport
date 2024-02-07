from scipy.stats import fisher_exact
from  collections import defaultdict
from functools import partial
import networkx as nx
import pickle
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import umap
import argparse
from sklearn.cluster import SpectralClustering, DBSCAN, HDBSCAN
from models.language_models import SentenceLukeJapanese
import collections
import plotly.graph_objs as go

color_names = """aliceblue, maroon, aqua, aquamarine, azure,
            ivory, bisque, black, lightyellow, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            beige, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lime, limegreen,
            linen, magenta, antiquewhite, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,blanchedalmond,
            royalblue, rebeccapurple, saddlebrown, salmon,
            sandybrown, seagreen, seashell, sienna, silver,
            skyblue, slateblue, slategray, slategrey, snow,
            springgreen, steelblue, tan, teal, thistle, tomato,
            turquoise, violet, wheat, white, whitesmoke,
            yellow, yellowgreen"""

# コンマで分割してリストに変換
color_list = [color.strip() for color in color_names.split(",")]

# 空白文字を削除し、空の要素を除去
colors = [color for color in color_list if color]

def get_target_col(df):
    if 'spot' in df.columns:
        return 'spot'
    elif 'food' in df.columns:
        return 'food'
    elif 'name' in df.columns:
        return 'name'
    
def get_correlation_matrix(spot, df_review, df_topics, conditions, suffix, extra_suffix,):
    # 各場所における各トピックの出現回数をカウントする
    save_name = args.df_topic_path.split('/')[-1].split('.')[0].replace('_merged', '')
    with open(f'../data/groups/group_{save_name}.pkl', 'rb') as f:
        groups = pickle.load(f)

    try:
        with open(f'../data/clustering/review/{spot}/{suffix}_{extra_suffix}_pos_cluster_sentence.pkl', 'rb') as f:
            d_pos = pickle.load(f)
    except FileNotFoundError:
        d_pos = {'clustering_labels': [], 'inds': []}
        
    try:
        with open(f'../data/clustering/review/{spot}/{suffix}_{extra_suffix}_neg_cluster_sentence.pkl', 'rb') as f:
            d_neg = pickle.load(f)
    except FileNotFoundError:
        d_neg = {'clustering_labels': [], 'inds': []}
        
    print(spot, groups[spot])
    pos_original_cluster_to_group = {-1:-1}
    for i,ks in enumerate(groups[spot]['pos']):
        for k in ks:
            pos_original_cluster_to_group[k] = i
    neg_original_cluster_to_group = {-1:-1}
    for i,ks in enumerate(groups[spot]['neg']):
        for k in ks:
            neg_original_cluster_to_group[k] = i
            
    d_pos['group_clustering_labels'] = [pos_original_cluster_to_group[l] for l in d_pos['clustering_labels']]
    d_neg['group_clustering_labels'] = [neg_original_cluster_to_group[l] for l in d_neg['clustering_labels']]
    
    
    target_col = get_target_col(df_review)
    df_target = df_review[df_review[target_col]==spot]
    #clustering_labels_pos = np.array(d_pos['clustering_labels'])
    #clustering_labels_neg = np.array(d_neg['clustering_labels'])
    clustering_labels_pos = np.array(d_pos['group_clustering_labels'])
    clustering_labels_neg = np.array(d_neg['group_clustering_labels'])
    if len(clustering_labels_pos) == 0 or len(clustering_labels_neg) == 0:return None, [], [], 0
    pos_inds = np.array(d_pos['inds']) #元のレビューのうち何番目か
    neg_inds = np.array(d_neg['inds']) #元のレビューのうち何番目か
    pos_cluster_num = max(np.unique(clustering_labels_pos))+1
    neg_cluster_num = max(np.unique(clustering_labels_neg))+1
    all_cluster_num = pos_cluster_num + neg_cluster_num
    correlation_matrix = np.zeros((all_cluster_num, all_cluster_num))
    total_topic_counts = np.zeros(all_cluster_num)
    total_num = 0
    for ind in list(df_target.index):
        skip_flg=False
        for k,v in conditions.items():
            if 'date' not in k:
                if df_target.loc[ind, k]!=v:
                    skip_flg=True
            elif k=='date_after':
                year =  df_review.loc[ind, 'year']
                month =  df_review.loc[ind, 'month']
                if pd.isna(year) or pd.isna(month):continue
                year, month = int(year), int(month)
                if year<2021:skip_flg=True
            elif k=='date_before':
                year =  df_review.loc[ind, 'year']
                month =  df_review.loc[ind, 'month']
                if pd.isna(year) or pd.isna(month):continue
                year, month = int(year), int(month)
                if year>=2021:skip_flg=True
        if skip_flg:continue
                
        pos_topics_tmp = clustering_labels_pos[pos_inds==ind]
        pos_topics_tmp = pos_topics_tmp[pos_topics_tmp!=-1]
        neg_topics_tmp = clustering_labels_neg[neg_inds==ind]
        neg_topics_tmp = neg_topics_tmp[neg_topics_tmp!=-1]+pos_cluster_num
        topic_joined = np.concatenate([pos_topics_tmp, neg_topics_tmp])
        if len(topic_joined):total_num+=1
        for t in topic_joined:
            total_topic_counts[t]+=1
        comb = combinations(topic_joined, 2)
        for t1, t2 in comb:
            correlation_matrix[t1, t2]+=1
            correlation_matrix[t2, t1]+=1
            
    return correlation_matrix, pos_cluster_num, total_topic_counts, total_num

def get_topics(spot, df_topics):
    topics = np.concatenate([df_topics[(df_topics['spot']==spot)&(df_topics['posneg']=='pos')]['topics'].values,
                        df_topics[(df_topics['spot']==spot)&(df_topics['posneg']=='neg')]['topics'].values])
    return topics
            
def visualize(topics, correlation_matrix, pos_cluster_num):
    plt.figure(figsize=(20, 20))
    topics = [topic.replace('<|im_end|>', '') for topic in topics]
    #if correlation_matrix.max()>1:
    #    correlation_matrix = correlation_matrix/correlation_matrix.max()
    # グラフの初期化
    G = nx.Graph()

    # ノードの追加
    for i, topic in enumerate(topics):
        G.add_node(topic+f'_{i}')

    # エッジ（共起関係）の追加。エッジの重みは共起の強さに基づく
    alpha = 0.1  # この値は0から1の範囲で調整する
    color = (0, 0, 0, alpha)  # 黒色のRGBA値
    for i, topic1 in enumerate(topics):
        for j, topic2 in enumerate(topics):
            if i < j:
                # correlation_matrix[i][j]が共起の強さを表す

                G.add_edge(topic1+f'_{i}', topic2+f'_{j}', weight=np.sqrt(correlation_matrix[i][j]), color=color)

    # ノードの位置の決定
    pos = nx.spring_layout(G)

    # ノードの色分け
    node_color = ['red' if i < pos_cluster_num else 'blue' for i in range(len(topics))]

    # エッジの太さの設定
    edge_width = [G[u][v]['weight'] for u, v in G.edges()]
    edge_color = [G[u][v]['color'] for u, v in G.edges()]

    # グラフの描画
    nx.draw(G, pos, with_labels=True, node_color=node_color, width=edge_width, font_size=15, node_size=500,edge_color=edge_color,font_family='IPAexGothic')
    plt.show()
    
def extract_diff_topics(spot, df_topics, df_review, condition1, condition2, suffix, extra_suffix, thresh=0.01):
    # ある場所におけるトピックの出現回数を男性, 女性でそれぞれカウントし，有意差があるかどうか調べる
    topics = get_topics(spot, df_topics)
    correlation_matrix, pos_cluster_num, total_counts, total_num = get_correlation_matrix(spot, df_review, df_topics, conditions={}, suffix=suffix, extra_suffix=extra_suffix)
    correlation_matrix_male, pos_cluster_num_male, total_counts_male, total_num_male = get_correlation_matrix(spot,df_review,  df_topics, conditions=condition1, suffix=suffix, extra_suffix=extra_suffix)
    correlation_matrix_female, pos_cluster_num_female, total_counts_female, total_num_female = get_correlation_matrix(spot, df_review, df_topics, conditions=condition2, suffix=suffix, extra_suffix=extra_suffix)
    male_topic = []
    female_topic = []
    male_topic_ind = []
    female_topic_ind = []
    for i, (mc, fc) in enumerate(zip(total_counts_male, total_counts_female)):
        table = [[mc, max(total_num_male-mc, 0)],
                [fc, max(total_num_female-fc, 0)]]
        odds_ratio, pval1 = fisher_exact(table)
        if pval1<thresh:
            if mc>fc:
                male_topic.append(topics[i])
                male_topic_ind.append(topics[i])
            else:
                female_topic.append(topics[i])
                female_topic_ind.append(topics[i])
    return male_topic, female_topic

def extract_diff_topics3(spot,df_topics, df_review, condition1, condition2,condition3, suffix, extra_suffix, thresh=0.01):
    topics = get_topics(spot, df_topics)
    correlation_matrix, pos_cluster_num, total_counts, total_num = get_correlation_matrix(spot, conditions={}, suffix=suffix, extra_suffix=extra_suffix)
    correlation_matrix_cond1, pos_cluster_num_cond1, total_counts_cond1, total_num_cond1 = get_correlation_matrix(spot, df_review, conditions=condition1, suffix=suffix, extra_suffix=extra_suffix)
    correlation_matrix_cond2, pos_cluster_num_cond2, total_counts_cond2, total_num_cond2 = get_correlation_matrix(spot, df_review, conditions=condition2, suffix=suffix, extra_suffix=extra_suffix)
    correlation_matrix_cond3, pos_cluster_num_cond3, total_counts_cond3, total_num_cond3 = get_correlation_matrix(spot, df_review, conditions=condition3, suffix=suffix, extra_suffix=extra_suffix)
    cond1_topic = []
    cond2_topic = []
    cond3_topic = []
    for i, (count1, count2, count3) in enumerate(zip(total_counts_cond1, total_counts_cond2, total_counts_cond3)):
        table12 = [[count1, max(total_num_cond1-count1, 0)],
                [count2, max(total_num_cond2-count2, 0)]]
        odds_ratio12, pval12 = fisher_exact(table12)
        table13 = [[count1, max(total_num_cond1-count1, 0)],
            [count3, max(total_num_cond3-count3, 0)]]
        odds_ratio13, pval13 = fisher_exact(table13)
        table23 = [[count2, max(total_num_cond2-count2, 0)],
            [count3, max(total_num_cond3-count3, 0)]]
        odds_ratio23, pval23 = fisher_exact(table23)
        if pval12<thresh and pval13<thresh and (count1/total_num_cond1)>(count2/total_num_cond2) and (count1/total_num_cond1)>(count3/total_num_cond3):
            cond1_topic.append(topics[i])
        elif pval12<thresh and pval23<thresh and (count2/total_num_cond2)>(count1/total_num_cond1) and (count2/total_num_cond2)>(count3/total_num_cond3):
            cond2_topic.append(topics[i])
        elif pval23<thresh and pval13<thresh and (count3/total_num_cond3)>(count1/total_num_cond1) and (count3/total_num_cond3)>(count2/total_num_cond2):
            cond3_topic.append(topics[i])

    return cond1_topic, cond2_topic, cond3_topic

def make_diff_topic_summary(spots, df_topics, df_review, condition1, condition2, suffix, extra_suffix, thresh=0.01):
    '''
    condition1とcondition2の有意差があったトピックを
    {'spot1': {'cond1': [topic1, topic2..], 'cond2': [topic3, topic4..]}}
    の形で得る
    '''
    cond1_topic_all = []
    cond2_topic_all = []
    all_topics = []
    diff_topics = defaultdict(partial(defaultdict, list))
    for spot in spots:
        cond1_topic, cond2_topic = extract_diff_topics(spot, df_topics, df_review, condition1, condition2, suffix, extra_suffix,thresh=thresh)
        diff_topics[spot][list(condition1.values())[0]] = cond1_topic
        diff_topics[spot][list(condition2.values())[0]] = cond2_topic
        cond1_topic_all+=cond1_topic
        cond2_topic_all+=cond2_topic
        
    return diff_topics,cond1_topic_all, cond2_topic_all

def make_diff_topic_summary3(spots, df_topics,df_review,  condition1, condition2, condition3,suffix, extra_suffix, thresh=0.01):
    cond1_topic_all = []
    cond2_topic_all = []
    cond3_topic_all = []
    diff_topics = defaultdict(partial(defaultdict, list))
    for spot in spots:
        cond1_topic, cond2_topic,cond3_topic = extract_diff_topics3(spot, df_topics, df_review, condition1, condition2, condition3, suffix, extra_suffix,thresh=thresh)
        diff_topics[spot][list(condition1.values())[0]] = cond1_topic
        diff_topics[spot][list(condition2.values())[0]] = cond2_topic
        cond1_topic_all+=cond1_topic
        cond2_topic_all+=cond2_topic
        
    return diff_topics,cond1_topic_all, cond2_topic_all, cond3_topic_all
    
def get_original_ind(df_topics, diff_topics):
    '''
    得られたトピックが元のdf_topicsのどこにあったか
    '''
    male_pos_inds, male_neg_inds = [], []
    female_pos_inds, female_neg_inds = [], []
    for spot, topics in diff_topics.items():
        male_topics = topics['男性']
        for topic  in male_topics:
            ind = df_topics[(df_topics['spot']==spot)&(df_topics['topics']==topic)].index[0]
            if df_topics.loc[ind, 'posneg']=='pos':
                male_pos_inds.append(ind)            
            elif df_topics.loc[ind, 'posneg']=='neg':
                male_neg_inds.append(ind)
            
        female_topics = topics['女性']
        for topic in female_topics:
            ind = df_topics[(df_topics['spot']==spot)&(df_topics['topics']==topic)].index[0]
            if df_topics.loc[ind, 'posneg']=='pos':
                female_pos_inds.append(ind)            
            elif df_topics.loc[ind, 'posneg']=='neg':
                female_neg_inds.append(ind)
    return male_pos_inds, male_neg_inds, female_pos_inds, female_neg_inds

def get_big_category(df_topics, min_cluster_size=20):
    feature_extractor=SentenceLukeJapanese("sonoisa/sentence-luke-japanese-base-lite", device='cuda')
    topics = df_topics['topics'].values
    topic_emb = []
    for i in range(len(topics)//100+1):
        emb_tmp = feature_extractor.encode(topics[i*100:min((i+1)*100, len(topics))], batch_size=len(topics[i*100:min((i+1)*100, len(topics))]))
        topic_emb.append(emb_tmp)
    topic_emb = np.concatenate(topic_emb)
    pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.05)
    emb_2d = pca.fit_transform(topic_emb)
    clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(emb_2d)
        
    colors_2d = [colors[min(clustering.labels_[i], len(colors)-1)] for i in range(len(topics))]

    trace = go.Scatter(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        mode='markers',
        text=topics,  # ホバーテキストとして文書の内容を使用
        hoverinfo='text',  # ホバー時にはテキストのみを表示fw∂
        marker=dict(color=colors_2d)
    )
    fig = go.Figure(trace)
    return fig, clustering.labels_

def get_gender_diff_topic(male_inds, female_inds, clustering_labels):
    male_cluster_labels, female_cluster_labels = [], []
    for i in male_inds:
        male_cluster_labels.append(clustering_labels[i])
    for i  in female_inds:
        female_cluster_labels.append(clustering_labels[i])
        
    male_counter = collections.Counter(male_cluster_labels)
    female_counter = collections.Counter(female_cluster_labels)
    male_counter = dict(male_counter)
    female_counter = dict(female_counter)
    male_counter = dict(sorted(male_counter.items(), key=lambda item: -item[1]))
    female_counter = dict(sorted(female_counter.items(), key=lambda item: -item[1]))
    return male_counter, female_counter

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_review_path', type=str, default='')
    parser.add_argument('--df_topic_path', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--extra_suffix', type=str, default='')
    args = parser.parse_args()
    df_review = pd.read_csv(args.df_review_path).reset_index()#df_review.reset_index()
    df_topics = pd.read_csv(args.df_topic_path,
                            names=['index', 'spot', 'posneg', 'cluster', 'topics'],
                            ).reset_index()
    # if df_topics.loc[0, 'spot']=='spot':
    #     df_topics = pd.read_csv(args.df_topic_path,)
    #print(df_topics.head())
    #for col in df_topics.columns:
        #print(col, df_topics[col])
    spots = list(df_topics['spot'].unique())
    if 'hotel' in args.df_review_path:
        diff_topics,male_topics, female_topics = make_diff_topic_summary(spots, df_topics, df_review, condition1={'gender': '男性'},condition2={'gender': '女性'}, 
                                                                        suffix=args.suffix, extra_suffix=args.extra_suffix,thresh=0.05)
    else:
        diff_topics,male_topics, female_topics = make_diff_topic_summary(spots, df_topics, df_review, condition1={'sex': '男性'},condition2={'sex': '女性'}, 
                                                                        suffix=args.suffix, extra_suffix=args.extra_suffix,thresh=0.05)
    #print(male_topics, female_topics)
    male_pos_inds, male_neg_inds, female_pos_inds, female_neg_inds = get_original_ind(df_topics, diff_topics)
    pos_fig, pos_clustering_labels = get_big_category(df_topics[df_topics['posneg']=='pos'], min_cluster_size=25)
    neg_fig, neg_clustering_labels = get_big_category(df_topics[df_topics['posneg']=='neg'], min_cluster_size=10)
    pos_index = list(df_topics[df_topics['posneg']=='pos'].index)
    neg_index = list(df_topics[df_topics['posneg']=='neg'].index)
    assert len(pos_index)==len(pos_clustering_labels)
    assert len(neg_index)==len(neg_clustering_labels)
    clustering_labels = np.zeros(len(df_topics))
    clustering_labels[pos_index] = pos_clustering_labels
    clustering_labels[neg_index] = neg_clustering_labels 
    male_pos_counter, female_pos_counter = get_gender_diff_topic(male_pos_inds, female_pos_inds, clustering_labels)
    male_neg_counter, female_neg_counter = get_gender_diff_topic(male_neg_inds, female_neg_inds, clustering_labels)
    #print('male pos')
    
    show_num=10
    for i,(k,v) in enumerate(male_pos_counter.items()):
        if v>=2:
            print(k, v)
            print(df_topics[df_topics['posneg']=='pos']['topics'][pos_clustering_labels==k].values[:show_num])
        if i==4:break
        
    print('male neg') 
    for i,(k,v) in enumerate(male_neg_counter.items()):
        if v>=2:
            print(k, v)
            print(df_topics[df_topics['posneg']=='neg']['topics'][neg_clustering_labels==k].values[:show_num])
        if i==4:break
        
    print('female pos')
    for i,(k,v) in enumerate(female_pos_counter.items()):
        if v>=2:
            print(k, v)
            print(df_topics[df_topics['posneg']=='pos']['topics'][pos_clustering_labels==k].values[:show_num])
        if i==4:break
        
    print('female neg')    
    for i,(k,v) in enumerate(female_neg_counter.items()):
        if v>=2:
            print(k, v)
            print(df_topics[df_topics['posneg']=='neg']['topics'][neg_clustering_labels==k].values[:show_num])
        if i==4:break