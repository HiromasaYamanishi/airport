import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, DBSCAN, HDBSCAN
import plotly.graph_objs as go
import umap
import os
import numpy as np
import argparse

def extract_feature(df:pd.DataFrame, pos_all: dict, condition:dict):
    '''
    input:
        df: 属性がついているreview_df
        pos_all: {1: [sentence1, sentence2], 2: [sentence3, sentence4.], 3:..}
    '''
    all_embs = []
    all_sents = []
    all_inds = []
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
            all_inds.append(ind)
            
    return all_embs, all_sents, all_inds

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

def filter_data(d, df, condition):
    for k,v in condition.items():
        df = df[df[k]==v]
    index = list(df.index)
    filtered_d = {k:v for k,v in d.items() if k in index}
    return filtered_d
    
def visualize(embs, sentences,spot,  save_path, args, save=False, clustering_2d=False, posneg='pos'):
    if posneg=='pos':
        pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.1)
    elif posneg=='neg':
        pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.01)
    print('len sentences', len(embs),)
    if len(embs)<10:return None, [-1 for _ in range(len(embs))]
    emb_2d = pca.fit_transform(embs)
    if posneg=='pos':
        min_cluster_size = max(min(max(len(sentences)//60, 20), len(sentences)//30), 5)
    elif posneg=='neg':
        min_cluster_size = max(min(max(len(sentences)//60, 20),len(sentences)//30), 5)
        
    if clustering_2d:
        clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(emb_2d)
    else:
        clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(emb)
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
        
    colors_2d = [colors[min(clustering.labels_[i], len(colors)-1)] for i in range(len(sentences))]
    
    trace = go.Scatter(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        mode='markers',
        text=sentences,  # ホバーテキストとして文書の内容を使用
        hoverinfo='text',  # ホバー時にはテキストのみを表示
        marker=dict(color=colors_2d, size=12)
    )
    fig = go.Figure(trace)
    # fig.update_layout(
    #     plot_bgcolor='white',  # プロットの背景色
    #     paper_bgcolor='white',  # 全体の背景色
    #     xaxis=dict(
    #         showgrid=True,  # x軸のグリッドを非表示にする
    #         zeroline=True  # x軸のゼロラインを非表示にする
    #     ),
    #     yaxis=dict(
    #         showgrid=True,  # y軸のグリッドを非表示にする
    #         zeroline=True  # y軸のゼロラインを非表示にする
    #     )
    # )

    if save:
        if not os.path.exists(f'../data/clustering/html/{spot}/'):
            os.makedirs(f'../data/clustering/html/{spot}/')
        fig.write_html(f'../data/clustering/html/{spot}/{save_path}_{posneg}.html')
    return fig, clustering.labels_

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--extra_suffix', type=str, default='')
    parser.add_argument('--div', type=int, default=1)
    parser.add_argument('--df_path', type=str, default='')
    #parser.add_argument('--n_neighbors', type=int, default=15)
    #parser.add_argument('--min_dist', type=float, default=0.1)
    args = parser.parse_args()
    #df = pd.read_pickle('/home/yamanishi/project/airport/src/data/review_all_period_.pkl')
    #df = df[df['pref']=='香川県'].reset_index()
    df = pd.read_csv(args.df_path)
    #df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/df_jalan_kagawa.csv')
    if 'spot' in df.columns:
        spots = np.unique(df['spot'].values)
        target_col = 'spot'
    elif 'hotel_name' in df.columns:
        spots = np.unique(df['hotel_name'].values)
        target_col = 'hotel_name'
    elif 'food' in df.columns:
        spots = np.unique(df['food'].values)
        target_col = 'food'

    pos_all_d = load_pos_data(args)
    neg_all_d = load_neg_data(args)
    kagawa_popular_spots = ['金刀比羅宮', '栗林公園', 'エンジェルロード', 'レオマリゾート', '丸亀城', '瀬戸大橋（香川県坂出市）', '寒霞渓ロープウェイ', '道の駅\u3000小豆島オリーブ公園', '屋島', '二十四の瞳映画村', '銭形砂絵「寛永通宝」', '史跡高松城跡（玉藻公園）', '国営讃岐まんのう公園', '新屋島水族館', '瀬戸大橋記念公園', 'さぬきこどもの国', '地中美術館', 'マルキン醤油記念館', '直島諸島', 'サンポート高松']
    kagawa_popular_spots = ['エンジェルロード']
    for spot in spots:
        save_path = args.suffix + '_'+args.extra_suffix
        print(spot, save_path, 'neg')
        #if os.path.exists(f'../data/clustering/{save_path}_neg.html'):continue
        emb, sents, inds = extract_feature(df, neg_all_d, {target_col: spot})
        fig, clustering_labels = visualize(emb, sents, spot, save_path, args,save=True, clustering_2d=True, posneg='neg')
        cluster_num = len([i for i in np.unique(clustering_labels) if i!=-1])
        print(cluster_num)
        d = {'sentence': sents,
             'clustering_labels': clustering_labels,
             'inds': inds}
        if not os.path.exists(f'../data/clustering/review/{spot}/'):
            os.makedirs(f'../data/clustering/review/{spot}/')
            
        with open(f'../data/clustering/review/{spot}/{save_path}_neg_cluster_sentence.pkl', 'wb') as f:
            pickle.dump(d, f)       
            
    for spot in spots:
        save_path = args.suffix + '_'+args.extra_suffix
        #print(save_path, 'pos')
        print(spot, save_path, 'pos')
        #if os.path.exists(f'../data/clustering/{save_path}_pos.html'):continue
        emb, sents, inds = extract_feature(df, pos_all_d, {target_col: spot})
        fig, clustering_labels = visualize(emb, sents, spot, save_path, args, save=True, clustering_2d=True, posneg='pos')
        cluster_num = len([i for i in np.unique(clustering_labels) if i!=-1])
        print('num_cluster', cluster_num)
        d = {'sentence': sents,
             'clustering_labels': clustering_labels,
             'inds': inds}
        
        with open(f'../data/clustering/review/{spot}/{save_path}_pos_cluster_sentence.pkl', 'wb') as f:
            pickle.dump(d, f)
            
            # emb_male, sents_male = extract_feature(df, neg_all_d, {'spot': spot, 'sex': '男性'})
        # fig, clustering_labels_male = visualize(emb_male, sents_male, save_path+'_male', args,save=True, clustering_2d=True, posneg='neg')
        # d = {'sentence': sents_male,
        #      'clustering_labels': clustering_labels_male}
        
        # with open(f'../data/clustering/review/{save_path}_male_neg_cluster_sentence.pkl', 'wb') as f:
        #     pickle.dump(d, f)
            
        # emb_female, sents_female = extract_feature(df, neg_all_d, {'spot': spot, 'sex': '女性'})
        # fig, clustering_labels_female = visualize(emb_female, sents_female, save_path+'_male', args, save=True, clustering_2d=True, posneg='neg')
        # d = {'sentence': sents_female,
        #      'clustering_labels': clustering_labels_female}
        
        # with open(f'../data/clustering/review/{save_path}_female_neg_cluster_sentence.pkl', 'wb') as f:
        #     pickle.dump(d, f)
            
        #emb_male, sents_male = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '男性'})
        # fig, clustering_labels_male = visualize(emb_male, sents_male, save_path+'_male', args, save=True, clustering_2d=True, posneg='pos')
        # d = {'sentence': sents_male,
        #      'clustering_labels': clustering_labels_male}
        
        # with open(f'../data/clustering/review/{save_path}_male_pos_cluster_sentence.pkl', 'wb') as f:
        #     pickle.dump(d, f)
            
        # emb_female, sents_female = extract_feature(df, pos_all_d, {'spot': spot, 'sex': '女性'})
        # fig, clustering_labels_female = visualize(emb_female, sents_female, save_path+'_male', args,save=True, clustering_2d=True, posneg='pos')
        # d = {'sentence': sents_female,
        #      'clustering_labels': clustering_labels_female}
        
        # with open(f'../data/clustering/review/{save_path}_female_pos_cluster_sentence.pkl', 'wb') as f:
        #     pickle.dump(d, f)
            
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
        

