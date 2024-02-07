import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, DBSCAN, HDBSCAN
import plotly.graph_objs as go
import umap
import os
import numpy as np
import argparse

def extract_feature(df, pos_all, condition):
    all_embs = []
    all_sents = []
    for ind, poss in pos_all.items():
        flag = False
        for row, val in condition.items():
            #print(ind, row)
            if df.loc[ind, row]!=val:
                flag=True
                
        if flag:continue
        for sent, emb in poss:
            all_sents.append(sent)
            all_embs.append(emb)
            
    return all_embs, all_sents

def load_pos_data(args):
    pos_all_d = {}
    pos_alls = []
    for i in range(args.div):
        with open(f'../data/review/pos_all_{args.suffix}_{i}.pkl', 'rb') as f:
            pos_all = pickle.load(f)
            pos_alls.append(pos_all)
            
    for pos_all in pos_alls:
        pos_all_d.update(pos_all)
    return pos_all_d

def load_neg_data(args):
    neg_all_d = {}
    neg_alls = []
    for i in range(args.div):
        with open(f'../data/review/neg_all_{args.suffix}_{i}.pkl', 'rb') as f:
            neg_all = pickle.load(f)
            neg_alls.append(neg_all)
            
    for neg_all in neg_alls:
        neg_all_d.update(neg_all)
    return neg_all_d

def visualize(embs, sentences, save_path, args, save=False, clustering_2d=False, posneg='pos'):
    pca=umap.UMAP(random_state=0, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    emb_2d = pca.fit_transform(embs)
    if clustering_2d:
        clustering = HDBSCAN(min_cluster_size=max(len(sentences)//50, 5)).fit(emb_2d)
    else:
        clustering = HDBSCAN(min_cluster_size=max(len(sentences)//50, 5)).fit(emb)
    color_names = """aliceblue, antiquewhite, aqua, aquamarine, azure,
                beige, bisque, black, blanchedalmond, blue,
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
                ivory, khaki, lavender, lavenderblush, lawngreen,
                lemonchiffon, lightblue, lightcoral, lightcyan,
                lightgoldenrodyellow, lightgray, lightgrey,
                lightgreen, lightpink, lightsalmon, lightseagreen,
                lightskyblue, lightslategray, lightslategrey,
                lightsteelblue, lightyellow, lime, limegreen,
                linen, magenta, maroon, mediumaquamarine,
                mediumblue, mediumorchid, mediumpurple,
                mediumseagreen, mediumslateblue, mediumspringgreen,
                mediumturquoise, mediumvioletred, midnightblue,
                mintcream, mistyrose, moccasin, navajowhite, navy,
                oldlace, olive, olivedrab, orange, orangered,
                orchid, palegoldenrod, palegreen, paleturquoise,
                palevioletred, papayawhip, peachpuff, peru, pink,
                plum, powderblue, purple, red, rosybrown,
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
        
    colors_2d = [colors[min(clustering.labels_[i], len(colors)-1)] for i in range(len(sentences))]
    
    trace = go.Scatter(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        mode='markers',
        text=sentences,  # ホバーテキストとして文書の内容を使用
        hoverinfo='text',  # ホバー時にはテキストのみを表示
        marker=dict(color=colors_2d)
    )
    fig = go.Figure(trace)
    if save:
        fig.write_html(f'../data/clustering/{save_path}_{posneg}.html')
    return fig, clustering.labels_

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--extra_suffix', type=str, default='')
    parser.add_argument('--div', type=int, default=1)
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.1)
    args = parser.parse_args()
    df= pd.read_pickle('/home/yamanishi/project/airport/src/data/review_all_period_.pkl')
    df_kagawa = df[df['pref']=='香川県'].reset_index()
    kagawa_popular_spots = ['金刀比羅宮', '栗林公園', 'エンジェルロード', 'レオマリゾート', '丸亀城', '瀬戸大橋（香川県坂出市）', '寒霞渓ロープウェイ', '道の駅\u3000小豆島オリーブ公園', '屋島', '二十四の瞳映画村', '銭形砂絵「寛永通宝」', '史跡高松城跡（玉藻公園）', '国営讃岐まんのう公園', '新屋島水族館', '瀬戸大橋記念公園', 'さぬきこどもの国', '地中美術館', 'マルキン醤油記念館', '直島諸島', 'サンポート高松']
    for model in ['luke', 'sentence']:
        args.suffix = 'incontext_kagawa_'+model
        pos_all_d = load_pos_data(args)
        neg_all_d = load_neg_data(args)
        result = []
        for posneg, d in enumerate([pos_all_d, neg_all_d]):
            for i,spot in enumerate(kagawa_popular_spots):
                save_path = spot+'_' + args.suffix + args.extra_suffix
                if spot not in kagawa_popular_spots[:5]:continue
                #if os.path.exists(f'../data/clustering/{save_path}_pos.html'):continue
                emb, sents = extract_feature(df_kagawa, d, {'spot': spot})
                
                for n_neighbors in [3,5,7, 10, 15]:
                    pca=umap.UMAP(random_state=0, n_neighbors=n_neighbors, min_dist=0.1)
                    emb_2d = pca.fit_transform(emb)
                    clustering = HDBSCAN(min_cluster_size=max(len(sents)//100, 5)).fit(emb_2d)
                    cluster_num = len([i for i in np.unique(clustering.labels_) if i!=-1])
                    #print(np.unique(clustering.labels_))
                    df = pd.DataFrame({'spot': [spot], 'posneg':[posneg], 'sentence_model': model, 'min_dist': [0.1], 'n_neighbors': [n_neighbors], 'algorithm':['auto'], 'min_cluster_size': [max(len(sents)//100, 5)], 'max_cluster_size': [None], 'cluster_num': [cluster_num]})
                    df.to_csv('../data/parameter_search/parameter_search.csv', mode='a', header=False)
                    
                for min_dist in [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1]:
                    pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=min_dist)
                    emb_2d = pca.fit_transform(emb)
                    clustering = HDBSCAN(min_cluster_size=max(len(sents)//100, 5)).fit(emb_2d)
                    cluster_num = len([i for i in np.unique(clustering.labels_) if i!=-1])
                    df = pd.DataFrame({'spot': [spot], 'posneg':[posneg], 'sentence_model': model, 'min_dist': [min_dist], 'n_neighbors': [10], 'algorithm':['auto'], 'min_cluster_size': [max(len(sents)//100, 5)], 'max_cluster_size': [None], 'cluster_num': [cluster_num]})
                    df.to_csv('../data/parameter_search/parameter_search.csv', mode='a', header=False)
                    
                for max_cluster_size in [10, 50, 100, 200, 500, None]:
                    pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.1)
                    emb_2d = pca.fit_transform(emb)
                    clustering = HDBSCAN(min_cluster_size=max(len(sents)//100, 5), max_cluster_size=max_cluster_size).fit(emb_2d)
                    cluster_num = len([i for i in np.unique(clustering.labels_) if i!=-1])
                    df = pd.DataFrame({'spot': [spot], 'posneg':[posneg], 'sentence_model': model, 'min_dist': [0.1], 'n_neighbors': [10], 'algorithm':['auto'], 'min_cluster_size': [max(len(sents)//100, 5)], 'max_cluster_size': [max_cluster_size], 'cluster_num': [cluster_num]})
                    df.to_csv('../data/parameter_search/parameter_search.csv', mode='a', header=False)
                
                for min_cluster_size in [ 3, 5, 10, 20, 50, 100]:
                    pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.1)
                    emb_2d = pca.fit_transform(emb)
                    clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=None).fit(emb_2d)
                    cluster_num = len([i for i in np.unique(clustering.labels_) if i!=-1])
                    df = pd.DataFrame({'spot': [spot], 'posneg':[posneg], 'sentence_model': model, 'min_dist': [0.1], 'n_neighbors': [10], 'algorithm':['auto'], 'min_cluster_size': [min_cluster_size], 'max_cluster_size': [None], 'cluster_num': [cluster_num]})
                    df.to_csv('../data/parameter_search/parameter_search.csv', mode='a', header=False)
                    
                for algo in ["auto", 'brute', 'kdtree', 'balltree']:
                    pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.1)
                    emb_2d = pca.fit_transform(emb)
                    clustering = HDBSCAN(min_cluster_size=max(len(sents)//100, 5), max_cluster_size=None, algorithm=algo).fit(emb_2d)
                    cluster_num = len([i for i in np.unique(clustering.labels_) if i!=-1])
                    df = pd.DataFrame({'spot': [spot], 'posneg':[posneg], 'sentence_model': model, 'min_dist': [0.1], 'n_neighbors': [10], 'algorithm':[algo], 'min_cluster_size': [max(len(sents)//100, 5)], 'max_cluster_size': [None], 'cluster_num': [cluster_num]})
                    df.to_csv('../data/parameter_search/parameter_search.csv', mode='a', header=False)

    exit()
    # for spot in kagawa_popular_spots:
    #     save_path = spot+'_male'
    #     if os.path.exists(f'../data/clustering/{save_path}.html'):continue
    #     emb, sents = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '男性'})
    #     fig = visualize(emb, sents, spot+'_male', save=True)
        
    for spot in kagawa_popular_spots:
        save_path = spot+'_male_2d'
        if os.path.exists(f'../data/clustering/{save_path}.html'):continue
        emb, sents = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '男性'})
        fig = visualize(emb, sents, spot+'_male_2d', save=True, clustering_2d=True)
        
    # for spot in kagawa_popular_spots:
    #     save_path = spot+'_female'
    #     if os.path.exists(f'../data/clustering/{save_path}.html'):continue
    #     emb, sents = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '女性'})
    #     fig = visualize(emb, sents, spot+'_female', save=True)
        
    for spot in kagawa_popular_spots:
        save_path = spot+'_female_2d'
        if os.path.exists(f'../data/clustering/{save_path}.html'):continue
        emb, sents = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '女性'})
        fig = visualize(emb, sents, spot+'_female_2d', save=True, clustering_2d=True)
        
    neg_all_d = load_neg_data()
    for spot in kagawa_popular_spots:
        save_path = spot + '_neg'
        if os.path.exists(f'../data/clustering/{save_path}.html'):continue
        emb, sents = extract_feature(df, neg_all_d, {'spot': spot})
        fig = visualize(emb, sents, save_path, save=True)

    for spot in kagawa_popular_spots:
        save_path = spot+'neg_2d'
        if os.path.exists(f'../data/clustering/{save_path}.html'):continue
        emb, sents = extract_feature(df, neg_all_d, {'spot': spot})
        fig = visualize(emb, sents, save_path, save=True, clustering_2d=True)
        
    # for spot in kagawa_popular_spots:
    #     save_path = spot+'_male'
    #     if os.path.exists(f'../data/clustering/{save_path}.html'):continue
    #     emb, sents = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '男性'})
    #     fig = visualize(emb, sents, spot+'_male', save=True)
        
    for spot in kagawa_popular_spots:
        save_path = spot+'neg_male_2d'
        if os.path.exists(f'../data/clustering/{save_path}.html'):continue
        emb, sents = extract_feature(df, neg_all_d, {'spot': spot, 'sex': '男性'})
        fig = visualize(emb, sents, save_path, save=True, clustering_2d=True)
        
    # for spot in kagawa_popular_spots:
    #     save_path = spot+'_female'
    #     if os.path.exists(f'../data/clustering/{save_path}.html'):continue
    #     emb, sents = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '女性'})
    #     fig = visualize(emb, sents, spot+'_female', save=True)
        
    for spot in kagawa_popular_spots:
        save_path = spot+'neg_female_2d'
        if os.path.exists(f'../data/clustering/{save_path}.html'):continue
        emb, sents = extract_feature(df, neg_all_d, {'spot': spot, 'sex': '女性'})
        fig = visualize(emb, sents, save_path, save=True, clustering_2d=True)
        

