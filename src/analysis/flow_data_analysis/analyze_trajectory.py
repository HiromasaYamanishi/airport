import pickle
from typing import List
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
from collections import defaultdict
from functools import partial
import folium
from analysis.utils.utils import get_color, style_function, make_kagawa_map, get_count_gdf
from analysis.utils.slide_writer import SlideWriter
from datetime import datetime

class TrajectoryAnalyzer:
    def __init__(self):
        '''
        読み込むもの:
            軌跡: 移動軌跡, ユーザー属性, 滞在時間, 移動の緯度経度
            施設情報: 場所の情報, 食べ物の情報，ホテルの情報
            カテゴリ情報: メッシュあたりのカテゴリのスコア
        '''
        with open('../data/context/trajectory.pkl', 'rb') as f:
            trajectory = pickle.load(f)
            
        with open('../data/kagawa_gdf.pkl', 'rb') as f:
            self.kagawa_gdf = pickle.load(f)
        
        self.mesh_city = dict(zip(self.kagawa_gdf['mesh'], self.kagawa_gdf['city']))
        self.mesh_region = dict(zip(self.kagawa_gdf['mesh'], self.kagawa_gdf['region']))
        self.user_attributes = trajectory['user_attribute']
        self.user_attributes_numpy = trajectory['user_attribute'].values
        self.trajectories = trajectory['trajectory']
        self.user_num = len(self.trajectories)
        self.transition_times = trajectory['transition_time']
        self.latlons = trajectory['latlon']
        self.df_spot_kagawa = pd.read_csv('../data/df_jalan_kagawa.csv').dropna(subset='review_count')
        self.df_hotel_kagawa = pd.read_csv('../data/hotel/hotel_kagawa.csv').dropna(subset='review_count')
        self.df_food_kagawa = pd.read_csv('../data/food_info_all.csv').dropna(subset='review_count')
        self.df_topic_poi =  pd.concat([self.df_spot_kagawa, self.df_food_kagawa,])
        self.df_poi_all = pd.concat([self.df_spot_kagawa, self.df_food_kagawa,self.df_hotel_kagawa])
        self.df_kagawa_in = pd.read_csv('../data/df_kagawa_in.csv')
        self.mesh_jenre_count = self.make_jenre_mesh_count()
        self.ages = ['15～17歳', '18～19歳',  '20～21歳', '22～29歳', 
                     '30～34歳', '35～39歳', '40～49歳','50～59歳','60～69歳',
                     '70歳～', ]
        
    def make_jenre_mesh_count(self):
        '''
        各メッシュあたりのカテゴリの重要度を計算
        '''
        mesh_jenre_count = defaultdict(partial(defaultdict, int))
        for mesh in self.kagawa_gdf['mesh'].unique():
            df_tmp = self.df_topic_poi[self.df_topic_poi['mesh']==mesh]
            df_hotel_tmp = self.df_hotel_kagawa[self.df_hotel_kagawa['mesh']==mesh]
            df_food_tmp = self.df_food_kagawa[self.df_food_kagawa['mesh']==mesh]
            df_tmp = df_tmp[df_tmp['review_count']>=20]
            review_counts = df_tmp['review_count'].values
            eps = 1e-6
            for jenre, review_count in zip(df_tmp['jenre'].values, review_counts):
                jenres = jenre.split(',')[:2]
                jenres = [j for j in jenres if j not in ['その他', 'その他名所', '観光施設・名所巡り']]
                jenres = [j for j in jenres if 'その他' not in j]
                #weight = np.log10(sum(review_counts))*np.log10(sum(review_counts))*\
                #review_count/(sum(review_counts)+eps)/(len(jenres)+eps)/np.log10(sum(review_counts)+sum(df_hotel_tmp['review_count'])+sum(df_food_tmp['review_count']))
                weight =sum(review_counts)*np.log10(sum(review_counts)+eps)*\
                review_count/(sum(review_counts)+eps)/(len(jenres)+eps)/(sum(review_counts)+sum(df_hotel_tmp['review_count'])+sum(df_food_tmp['review_count'])+eps)
                #print(weight, (sum(review_counts)+sum(df_hotel_tmp['review_count'])+sum(df_food_tmp['review_count'])+eps), sum(review_counts)+eps, (len(jenres)+eps),np.log10(sum(review_counts)+eps))
                for j in jenres:
                    mesh_jenre_count[mesh][j]+=weight
        return mesh_jenre_count
        
    def filter_trajectory(self, target_user_attributes=[], target_cities=[], target_regions=[], target_meshs=[]):
        '''
        target_user_attributes: 指定したユーザーの属性が全て含まれる必要がある
        target_cities: 指定する都市のうちどれか含まれていればよい
        target_regions: 指定した地域のうちどれか含まれていれば良い
        target_meshs: 指定したメッシュのうちどれかが含まれていれば良い
        '''
        valid_index = []
        for i, trajectory in enumerate(self.trajectories):
            user_attribute = self.user_attributes_numpy[i]
            if len(target_user_attributes):
                skip_flg=True
                age_attributes = [i for i in target_user_attributes if '歳' in i]
                other_attributes = [i for i in target_user_attributes if '歳' not in i]
                for target_user_attribute in age_attributes:
                    if target_user_attribute in  user_attribute:skip_flg = False
                    
                for target_user_attribute in other_attributes:
                    if target_user_attribute not in user_attribute:skip_flg=True
                if skip_flg:continue

            if len(target_cities):
                skip_flg = True
                for mesh in trajectory:
                    if mesh not in self.mesh_city:continue
                    if self.mesh_city[mesh] in target_cities:
                        skip_flg = False
                if skip_flg:continue

            if len(target_regions):
                skip_flg = True
                for mesh in trajectory:
                    if mesh not in self.mesh_region:continue
                    if self.mesh_region[mesh] in target_regions:
                        skip_flg = False
                if skip_flg:continue
                
            if len(target_meshs):
                skip_flg = True
                for mesh in trajectory:
                    #if mesh==target_meshs[0]:
                        #print(user_attribute)
                    if mesh in target_meshs:
                        skip_flg = False
                if skip_flg:continue

            valid_index.append(i)
        
        return valid_index
    
    @staticmethod
    def show_pie(column, df_flow, save_name=None, kwargs={}):
        '''
        dfの指定して列のpie_plotを行う
        '''
        plt.figure()
        plt.rcParams['font.size'] = 15
        if column=='性別':
            colors = {'男性': 'lightgreen', '女性': 'red', float('nan'): 'green'}
        elif column=='年齢':
            colors = {'50～59歳':'#FF0000', '35～39歳':'#0000FF', '22～29歳':'#008000' , '70歳～':'#00FFFF', 
                      '60～69歳':'#FF00FF', '18～19歳':'#FFFF00','30～34歳':'#FFDAB9' , 
                      '40～49歳':'#F5FFFA', '20～21歳':'#FFA500', '15～17歳':'#FFB6C1'}

        default_colors = plt.get_cmap('tab20').colors[::-1]  # matplotlibのタブカラーマップからデフォルトの色を取得
        df_flow[column] = df_flow[column][df_flow[column]!='nan']
        color_list = [colors[label] if label in colors else default_colors[i % len(default_colors)] for i, label in enumerate(df_flow[column].value_counts().index)]
        plt.pie(df_flow[column].value_counts(dropna=True), labels=df_flow[column].value_counts(dropna=True).index, colors=color_list,autopct='%1.1f%%', **kwargs)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(f'../data/result/user_atttribute_{save_name}.jpg')
        
    def analyze_user_attributes(self, target_cities, target_regions, target_meshs, save_name):
        '''
        フィルタリングを行った後pie_plotを行う
        '''
        target_indexes = self.filter_trajectory(
                                                target_cities=target_cities, 
                                                target_regions=target_regions, 
                                                target_meshs=target_meshs)
        user_target_df = self.user_attributes.loc[target_indexes]
        TrajectoryAnalyzer.show_pie('性別', user_target_df, save_name+'_gender')
        TrajectoryAnalyzer.show_pie('年齢', user_target_df, save_name+'_age')
        print(len(user_target_df), user_target_df)
        
    def show_trajectory(self, target_user_attributes,target_cities, target_regions, target_meshs, save_name):
        target_indexes = self.filter_trajectory(
                                        target_user_attributes,
                                        target_cities=target_cities, 
                                        target_regions=target_regions, 
                                        target_meshs=target_meshs)
        target_latlons = [self.latlons[i] for i in target_indexes]
        target_user_attributes =  [self.user_attributes_numpy[i] for i in target_indexes]
        #print(target_user_attributes)
        
        kagawa_gdf = get_count_gdf(self.df_kagawa_in, self.kagawa_gdf)
        m = make_kagawa_map(kagawa_gdf, style_function, df_jalan=self.df_spot_kagawa, df_jalan_food=self.df_food_kagawa,
                            df_hotel=self.df_hotel_kagawa,add_icon=True, add_icon_food=True, add_icon_hotel=True)
        for i,(target_ind,traj) in enumerate(zip(target_indexes, target_latlons)):
            user_attribute = target_user_attributes[i]
            if datetime.strptime(user_attribute[0], '%Y-%m-%d')>datetime.strptime('2023-04-14', '%Y-%m-%d'):
                print(user_attribute[0])
                folium.PolyLine(
                    locations=traj[~np.isnan(traj).any(axis=1)],     
                    color=get_color(target_ind, vmin=0, vmax=self.user_num,),
                    ).add_to(m)
                
        m.save(f'../data/html/filter/{save_name}.html')
        
    def analyze_freq_transit_pattern(self, target_user_attributes,target_cities, target_regions, target_meshs, save_name):
        target_indexes = self.filter_trajectory(
                                        target_user_attributes=target_user_attributes,
                                        target_cities=target_cities, 
                                        target_regions=target_regions, 
                                        target_meshs=target_meshs)
        target_traj = [self.trajectories[i] for i in target_indexes]
        trans_pattern = defaultdict(int)
        for traj in target_traj:
            for i in range(len(traj)-1):
                mesh1, mesh2 = traj[i], traj[i+1]
                if mesh1==mesh2:continue
                trans_pattern[(mesh1, mesh2)]+=1
                
        trans_pattern = dict(sorted(trans_pattern.items(), key=lambda item: item[1], reverse=True))
        for i,(k,v) in enumerate(trans_pattern.items()):
            from_pois = self.df_poi_all[self.df_poi_all['mesh']==k[0]][['spot_name', 'food_name','hotel_name','review_count']]
            to_pois = self.df_poi_all[self.df_poi_all['mesh']==k[1]][['spot_name', 'food_name','hotel_name', 'review_count']]
            print('from:', k[0], from_pois[from_pois['review_count']>=20].values)
            print('to:', k[1], to_pois[to_pois['review_count']>=20].values)
            print('count:', v)
            if i==20:break
        #print(trans_pattern)
        
    def analyze_freq_visit_jenre(self, target_user_attributes,target_cities, target_regions, target_meshs, save_name):
        '''
        よく訪れるジャンルを分析する
        '''
        target_indexes = self.filter_trajectory(
                                        target_user_attributes=target_user_attributes,
                                        target_cities=target_cities, 
                                        target_regions=target_regions, 
                                        target_meshs=target_meshs)
        target_traj = [self.trajectories[i] for i in target_indexes]
        
        cats_target = defaultdict(float)
        cats_target_mesh = defaultdict(partial(defaultdict, float))
        cats_not_target = defaultdict(float)
        cats_not_target_mesh = defaultdict(partial(defaultdict, float))
        print(len(target_traj))
        for traj in target_traj:
            for mesh in traj:
                #mesh = int(float(mesh[0]))
                if mesh not in self.mesh_city:continue
                elif (len(target_cities) and self.mesh_city[mesh] in target_cities)\
                    or (len(target_meshs) and mesh in target_meshs):
                    for jenre, score in self.mesh_jenre_count[mesh].items():
                        cats_target[jenre]+=score
                        cats_target_mesh[jenre][mesh]+=score
                else:
                    for jenre, score in self.mesh_jenre_count[mesh].items():
                        cats_not_target[jenre]+=score
                        cats_not_target_mesh[jenre][mesh]+=score
                        
        cats_target = dict(sorted(cats_target.items(), key=lambda item: item[1], reverse=True)[:7])
        top_cats = list(cats_target.keys())[:7]
        print('target')
        for i, top_cat in enumerate(top_cats):
            print(f'top{i} category: {top_cat}')
            cat_mesh = cats_target_mesh[top_cat]
            cat_mesh = dict(sorted(cat_mesh.items(), key=lambda item: item[1], reverse=True)[:3])
            cat_mesh = list(cat_mesh.keys())[:3]
            for mesh in cat_mesh:
                df_tmp = self.df_topic_poi[self.df_topic_poi['mesh']==mesh]
                print(df_tmp['jenre'])
                df_tmp = df_tmp[df_tmp['jenre'].str.contains(top_cat)]
                df_tmp = df_tmp[df_tmp['review_count']>=50]
                print(df_tmp[['spot_name', 'food_name', 'review_count']])
                
        print('not target')
        cats_not_target = dict(sorted(cats_not_target.items(), key=lambda item: item[1], reverse=True)[:8])
        top_not_cats = list(cats_not_target.keys())
        spots = []
        for i, top_cat in enumerate(top_not_cats):
            if top_cat=='和食':continue
            print(f'top{i} category: {top_cat}')
            cat_not_mesh = cats_not_target_mesh[top_cat]
            cat_not_mesh = dict(sorted(cat_not_mesh.items(), key=lambda item: item[1], reverse=True)[:3])
            cat_not_mesh = list(cat_not_mesh.keys())
            cat_spots =''
            for mesh in cat_not_mesh:
                df_tmp = self.df_topic_poi[self.df_topic_poi['mesh']==mesh]
                df_tmp = df_tmp[df_tmp['jenre'].str.contains(top_cat)]
                df_tmp = df_tmp[df_tmp['review_count']>=20].reset_index()
                name =df_tmp.loc[0, 'spot_name'] if pd.isna(df_tmp.loc[0, 'food_name']) else df_tmp.loc[0, 'food_name'] 
                city = df_tmp.loc[0, 'city']
                cat_spots+=f'{name} ({city}),'
                print(df_tmp[['spot_name', 'food_name', 'review_count', 'city']].values)
                #spot = df_tmp[['spot_name', 'food_name', 'review_count']].values[0][1]
                #if pd.isna(spot):continue
                #spots.append(spot)
            spots.append([top_cat, cat_spots])

        return np.array(spots)

                
        
        
        
if __name__ == '__main__':
    trajectory_analyzer = TrajectoryAnalyzer()
    trajectory_analyzer.show_trajectory(target_user_attributes=[],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51332794],
                                                 save_name='ヤドン')
    exit()
    spot_yadon = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=[],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51332794],
                                                 save_name='ヤドン')
    print(spot_yadon)
    spot_takagi = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=[],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51345185],
                                                 save_name='高木さん')
    print(spot_takagi)
    print(spot_yadon.shape)
    print(spot_takagi.shape)
    slide_writer = SlideWriter()
    row_names = [str(i+1) for i in range(8)]
    col_names = ['カテゴリ', '場所']
    slide_writer.write_tables([spot_yadon], [row_names], [col_names])
    slide_writer.save('../data/slide/yadon_rank.pptx')
    exit()
    slide_writer = SlideWriter()
    row_names = [str(i+1) for i in range(8)]
    col_names = ['カテゴリ', '場所']
    slide_writer.write_tables([spot_takagi], [row_names], [col_names])
    slide_writer.save('../data/slide/takagi_rank.pptx')
    exit()
    m1_udons = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=['男性', '15～17歳', '18～19歳',  '20～21歳', '22～29歳', '30～34歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[],
                                                 save_name='M1')
    m2_udons = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=['男性', '35～39歳', '40～49歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[],
                                                 save_name='M2')
    m3_udons = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=['男性', '50～59歳','60～69歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[],
                                                 save_name='M3')
    f1_udons = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=['女性', '15～17歳', '18～19歳',  '20～21歳', '22～29歳', '30～34歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[],
                                                 save_name='F1')
    f2_udons = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=['女性', '35～39歳', '40～49歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[],
                                                 save_name='F2')
    f3_udons = trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=['女性', '50～59歳','60～69歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[],
                                                 save_name='F3')
    udon_ranks = np.stack([m1_udons, m2_udons, m3_udons, f1_udons, f2_udons, f3_udons])
    print(udon_ranks)
    with open('../data/result/udon_ranks.pkl', 'wb') as f:
        pickle.dump(udon_ranks, f)
    slide_writer = SlideWriter()
    row_names = ['M1', 'M2', 'M3', 'F1', 'F2', 'F3']
    col_names = [f'ランク{i+1}' for i in range(10)]
    slide_writer.write_tables([udon_ranks], [row_names], [col_names])
    slide_writer.save('../data/slide/udon_rank.pptx')
        
    
    exit()
    trajectory_analyzer.analyze_freq_transit_pattern(target_user_attributes=[],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51345185],
                                                 save_name='高木さん')
    exit()
    trajectory_analyzer.show_trajectory(target_user_attributes=['男性', '22~29歳', '30～34歳', '35～39歳','40～49歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51345185, 51345163],
                                                 save_name='male_M1_takagi')
    trajectory_analyzer.show_trajectory(target_user_attributes=['女性', '50～59歳', '60～69歳'],
                                            target_cities=[], 
                                            target_regions=[],
                                            target_meshs=[51345185, 51345163],
                                            save_name='female_F3_takagi')
    exit()
    trajectory_analyzer.show_trajectory(target_user_attributes=['女性', '22~29歳', '30～34歳', '35～39歳','40～49歳'],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51345185],
                                                 save_name='male_F1_takagi')
    exit()
    trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=[],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51332794],
                                                 save_name='ヤドン')
    trajectory_analyzer.analyze_freq_visit_jenre(target_user_attributes=[],
                                                target_cities=[], 
                                                 target_regions=[],
                                                 target_meshs=[51345185],
                                                 save_name='高木さん')

    # trajectory_analyzer.analyze_user_attributes(target_cities=[], 
    #                                             target_regions=[],
    #                                             target_meshs=[51332794],
    #                                             save_name='ヤドン')
    # trajectory_analyzer.analyze_user_attributes(target_cities=[], 
    #                                             target_regions=[],
    #                                             target_meshs=[51345185],
    #                                             save_name='高木さん')
            
            
                
                
        
        

    