import pandas as pd
import pandas as pd
import json
import pickle
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplleaflet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ipyleaflet import Map, GeoJSON, basemaps, basemap_to_tiles
import lightgbm as lgb
from matplotlib.animation import PillowWriter
import matplotlib
import functools
import shap
import cartopy.crs as ccrs
import re
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.animation as animation
from PIL import Image
from collections import defaultdict
from shapely.geometry import Point
import tilemapbase
import folium
from folium import GeoJson
import japanize_matplotlib
from scipy import stats
from typing import Dict, List
import datetime
import os
import asyncio
from pyppeteer import launch
from PIL import Image
from datetime import datetime, timedelta
from geopy.distance import geodesic
from datetime import datetime, timedelta
from scipy.optimize import linear_sum_assignment
import numpy as np

def get_trajectory_gdf(df_flow, kagawa_gdf):
    #print(df_flow['mesh'])
    df_flow_tmp = df_flow[['mesh', 'diff_time']]
    df_flow_tmp = df_flow_tmp.sort_values('diff_time')
    df_flow_tmp = df_flow_tmp.drop_duplicates('mesh', keep='last')
    gdf_tmp = kagawa_gdf.copy()
    #print(mesh_visit_count_df, gdf_tmp)
    gdf_tmp = gdf_tmp.merge(df_flow_tmp, on='mesh', how='left')
    gdf_tmp['diff_time']=gdf_tmp['diff_time'].fillna(-25)
    return gdf_tmp

def calculate_iou(line1, line2):
    # line1とline2の始点と終点を取得
    start1, end1 = line1
    start2, end2 = line2
    # 両方の線分の長さを計算
    length1 = end1 - start1
    length2 = end2 - start2
    
    # 交差する部分の長さを計算
    intersection = max(0, min(end1, end2) - max(start1, start2))
    
    # IOUを計算
    union = length1 + length2 - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

def calc_cost(v, beh_time, lat, lon):
    if len(v)==0:
        return 1000
    else:
        prev_beh_time, prev_lat, prev_lon = v[-1][1:]
        dist = geodesic((prev_lat, prev_lon), (lat, lon)).kilometers
        time_diff = (beh_time-prev_beh_time)/timedelta (hours=1)
        return time_diff*100 + dist
    
def get_mesh(ind_right):
    if ind_right is None or pd.isna(ind_right) or np.isnan(ind_right):
        return None
    else:
        return kagawa_gdf['mesh'][ind_right]
    
def get_mesh_spot(df, gdf):
    '''
    input:
        df: DataFrame which have longitude and latitude
        gdf: GeoDataFrame which have mesh type geometry
    output:
        df with mesh information each column of df is in
    '''    
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    df['geometry'] = geometry
    gdf_points = gpd.GeoDataFrame(df, geometry='geometry')
    gdf['mesh'] = gdf['code'].astype(int)
    # 空間結合を行う
    result = gpd.sjoin(gdf_points, gdf, how='left', op='within')
    # 判定結果が含まれる列を取得
    result = result[['latitude', 'longitude', 'geometry', 'index_right']]  # 必要な列を選択

    # 必要な列を含む DataFrame に変換
    result_df = result.drop(columns='geometry')
    df['mesh'] = result_df['index_right'].apply(get_mesh)
    return df
    
def split_trajectory(v, df_kagawa, max_value):
    df_tmp = df_kagawa.loc[v]
    vs = [[] for _ in range(max_value)]
    for diff_time in df_tmp['diff_time'].unique():
        df_time = df_tmp[df_tmp['diff_time']==diff_time]
        cost_function = np.zeros((len(df_time), max_value))
        for i,ind in enumerate(df_time.index):
            for j in range(max_value):
                mesh = df_time.loc[ind, 'mesh']
                beh_time = df_time.loc[ind, 'beh_time']
                lat = mesh_to_lat[mesh]
                lon = mesh_to_lon[mesh]
                cost_function[i, j] = calc_cost(vs[j], beh_time, lat, lon)
        row_ind, col_ind = linear_sum_assignment(cost_function)
        for ind, ci in zip(df_time.index, col_ind):
            vs[ci].append((ind, df_time.loc[ind, 'beh_time'], mesh_to_lat[df_time.loc[ind, 'mesh']], mesh_to_lon[df_time.loc[ind, 'mesh']]))
            
    #print(df_time.loc[ind, ['mesh', 'beh_time']])
    vs = [[v_[0] for v_ in v] for v in vs]
    return vs

def make_trajectory_story(df):
    df = df.sort_values('diff_time')
    text = ''
    texts = []
    for diff_time, beh_time_hour, mesh in zip(df['diff_time'], df['beh_time_hour'], df['mesh']):
        text_hour = []
        text+=f'{int(beh_time_hour)}時(高松空港到着{diff_time}時間後)です\n'
        text_hour.append(f'{int(beh_time_hour)}時(高松空港到着{diff_time}時間後)です')
        spots = ','.join(mesh_to_spot[mesh])
        foods = ','.join(mesh_to_food[mesh])
        hotels = ','.join(mesh_to_hotel[mesh])
        num_hotels = len(mesh_to_hotel[mesh])
        text+=f'そこにある観光地は{spots}, 飲食店は{foods}です ホテルは{hotels}です'
        text_hour.append(f'そこにある観光地は{spots}, 飲食店は{foods}です ホテルは{hotels}です')
        texts.append((text_hour, mesh))
    return texts

def aggregate_story(story):
    '''
    texts: [(text_hour, mesh)の列]
    '''
    prev_mesh, groups, group_tmp, texts, prev_text, meshs = -1, [], [], [], None, []
    for text, mesh in story:
        if mesh==prev_mesh or prev_mesh==-1:
            group_tmp.append(text[0].replace('です', ''))
        elif mesh!=prev_mesh:
            groups.append(group_tmp)
            texts.append(prev_text)
            meshs.append(mesh)
            group_tmp = [text[0].replace('です', '')]
        prev_mesh = mesh 
        prev_text = text[1]
    if len(group_tmp):
        groups.append(group_tmp)
        texts.append(prev_text)
        meshs.append(mesh)
        
    grouped_story = ''
    grouped_story_list = []
    #grouped_story = []
    for group, text, mesh in zip(groups, texts, meshs):
        #grouped_story.append([f'{group[0]}から{group[-1]}までは{mesh}にいて', text])
        grouped_story += f'{group[0]}から{group[-1]}までは{mesh_to_city[mesh]}({mesh})にいて{text}'
        grouped_story_list.append(f'{group[0]}から{group[-1]}までは{mesh_to_city[mesh]}({mesh})にいて{text}')
        grouped_story += '\n'
        
    return grouped_story, grouped_story_list

def get_context(start, end, spots, foods, hotels):
    #print(start, end)
    if end-start>3 and calculate_iou((0, 8), (start, end))>0.5:
        # 夜に長時間滞在していれば宿泊
        return '宿泊'
    elif (len(spots)==0 or spots[0][1]<10) and (len(foods)==0 or foods[0][1]<10) and (len(hotels)==0 or hotels[0][1]<10):
        # 観光地・ホテル・飲食店がなければ移動
        return '移動'
    elif 11 in list(range(start, end+1)) or 12 in list(range(start, end+1)) or 13 in list(range(start, end+1)):
        #
        if 0<=(end-start) and (end-start)<=2 and len(foods) and foods[0][1]>10:
            # 昼の時間で飲食店があれば食事
            return '食事'
        elif end-start>2 and len(foods) and foods[0][1]>10 and len(spots) and spots[0][1]>10:
            # 滞在時間が長く飲食店も観光地もあれば食事と観光
            return '食事と観光'
        elif len(spots) and spots[0][1]>30:
            # 上に当てはまらず観光地があれば観光
            return '観光'
    elif 18 in list(range(start, end+1)) or 19 in list(range(start, end+1)) or 20 in list(range(start, end+1)):
        # 同様
        if 0<=(end-start) and (end-start)<=2 and len(foods) and foods[0][1]>10:
            # 昼の時間で飲食店があれば食事
            return '食事'
        elif (end-start)>2 and len(foods) and len(spots) and foods[0][1]>10 and spots[0][1]>10:
            # 滞在時間が長く飲食店も観光地もあれば食事と観光
            return '食事と観光'
        elif len(spots) and spots[0][1]>30:
            # 上に当てはまらず観光地があれば観光
            return '観光'
    elif (len(spots) and spots[0][1]>30):
        # 観光地があれば観光
        return '観光'
    else:
        # 上記に当てはまらなければ移動
        return '移動'
    
def get_mesh(ind_right):
    if ind_right is None or pd.isna(ind_right) or np.isnan(ind_right):
        return None
    else:
        return kagawa_gdf['mesh'][ind_right]

def attach_context(grouped_story_list):
    contexts_all = []
    for grouped_story in grouped_story_list:
        contexts = [grouped_story]
        pattern = r'(\d+)時'  # 数字(\d+)の後に"時"が続くパターンをマッチング
        matches = re.findall(pattern, grouped_story)
        start = int(matches[0])
        end = int(matches[2])
        
        pattern = r'観光地は([^,]*)です, 飲食店は([^$]*)です'  # パターンマッチング
        pattern_spot = r'観光地は(.*?)(?=飲食店は)'
        pattern_food = r'飲食店は(.*?)(?=です)'
        pattern_hotel = r'ホテルは(.*?)(?=です)'
        matches_spot = re.findall(pattern_spot, grouped_story)

        if matches_spot:
            sightseeing_spot = matches_spot[0]
            spots = sightseeing_spot.split(',')
            spots = [s for s in spots if s not in [' ', '']]
            spots = [(s, spot_to_review[s]) for s in spots]
            spots.sort(key=lambda x: x[1], reverse=True)
            print(spots[:3]) # "高松市レンタサイクル 丸亀町グリーン 高松市美術館"
            contexts.append(spots[:5])
        else:
            print("No match")
            
        matches_food = re.findall(pattern_food, grouped_story)

        if matches_food:
            sightseeing_food = matches_food[0]
            foods = sightseeing_food.split(',')
            foods = [s for s in foods if s not in [' ', '']]
            foods = [(s, food_to_review[s]) for s in foods]
            foods.sort(key=lambda x: x[1], reverse=True)
            print(foods[:3])# "高松市レンタサイクル 丸亀町グリーン 高松市美術館"
            contexts.append(foods[:3])
        else:
            print("No match")
            
        matches_hotel = re.findall(pattern_hotel, grouped_story)
        if matches_hotel:
            sightseeing_hotel = matches_hotel[0]
            hotels = sightseeing_hotel.split(',')
            hotels = [s for s in hotels if s not in [' ', '']]
            hotels = [(s, hotel_to_review[s]) for s in hotels if s in hotel_to_review]
            hotels.sort(key=lambda x: x[1], reverse=True)
            print(hotels[:3])# "高松市レンタサイクル 丸亀町グリーン 高松市美術館"
            contexts.append(hotels[:5])
        else:
            print("No match")
            
        context = get_context(start, end, spots, foods, hotels, )
        contexts.append(context)
        #print(context)
        contexts_all.append(contexts)
        
    return contexts_all
        
count=0
start_date = '2019-04-01'
end_date = '2023-09-30'
df_flow = pd.read_csv('../data/IPOCA-report_behavior-airport_takamatsu_rev.csv',)
df_flow['pas_time'] = pd.to_datetime(df_flow['pas_time'])
df_flow['beh_time'] = pd.to_datetime(df_flow['beh_time'])
df_flow['pas_time_month'] = df_flow['pas_time'].dt.month
df_flow['pas_time_year'] = df_flow['pas_time'].dt.year
df_flow['beh_time_month'] = df_flow['beh_time'].dt.month
df_flow['beh_time_year'] = df_flow['beh_time'].dt.year
df_flow['beh_time_hour'] = df_flow['beh_time'].dt.hour

df_jalan = pd.read_csv('../data/experience_light.csv')
df_jalan_kagawa = df_jalan[df_jalan['prefecture']=='香川県']
df_jalan_food = pd.read_csv('/home/yamanishi/project/airport/src/data/food_info_all.csv')
df_jalan_food=df_jalan_food[df_jalan_food['review_count']>=10]
with open("../data/shp/37kagawa1km.geojson") as f:
    kagawa = json.load(f)
kagawa_gdf = gpd.GeoDataFrame.from_features(kagawa['features'])
kagawa_gdf['mesh'] = kagawa_gdf['code'].astype(int)

df_jalan_food = get_mesh_spot(df_jalan_food, kagawa_gdf)
df_jalan_kagawa = get_mesh_spot(df_jalan_kagawa, kagawa_gdf)



mesh_to_spot = defaultdict(list)
for mesh, spot, rc in zip(df_jalan_kagawa['mesh'], df_jalan_kagawa['spot_name'], df_jalan_kagawa['review_count']):
    if rc>20:
        mesh_to_spot[mesh].append(spot)
mesh_to_food = defaultdict(list)
for mesh, food, rc in zip(df_jalan_food['mesh'], df_jalan_food['food_name'], df_jalan_food['review_count']):
    if rc>20:
        mesh_to_food[mesh].append(food) 
mesh_to_hotel = defaultdict(list)

df_hotel = pd.read_csv('/home/yamanishi/project/airport/src/data/hotel_info.csv')
df_hotel_kagawa = df_hotel[df_hotel['prefecture']=='香川県']
df_hotel_kagawa = get_mesh_spot(df_hotel_kagawa, kagawa_gdf)
for mesh, hotel, rc in zip(df_hotel_kagawa['mesh'], df_hotel_kagawa['hotel_name'], df_jalan_food['review_count']):
    if rc>=5:
        mesh_to_hotel[mesh].append(hotel) 
kagawa_gdf_region = gpd.read_file('../data/shp/N03-19_37_190101.shp')
result_gdf = gpd.overlay(kagawa_gdf, kagawa_gdf_region, how='identity')
result_gdf = result_gdf.drop_duplicates('code',keep='first' )
kagawa_gdf = pd.merge(kagawa_gdf, result_gdf[['code', 'N03_004']], on='code')
kagawa_gdf['city'] = kagawa_gdf['N03_004']

mesh_to_city = dict(zip(kagawa_gdf['mesh'], kagawa_gdf['city']))
mesh_to_city = defaultdict(lambda: None, mesh_to_city)
center_points = kagawa_gdf.geometry.centroid
kagawa_gdf['longitude'] = center_points.x
kagawa_gdf['latitude'] = center_points.y
mesh_to_lon = dict(zip(kagawa_gdf['mesh'], kagawa_gdf['longitude']))
mesh_to_lat = dict(zip(kagawa_gdf['mesh'], kagawa_gdf['latitude']))

spot_to_review = dict(zip(df_jalan_kagawa['spot_name'], df_jalan_kagawa['review_count']))
food_to_review = dict(zip(df_jalan_food['food_name'], df_jalan_food['review_count']))
hotel_to_review = dict(zip(df_hotel_kagawa['hotel_name'], df_jalan_food['review_count']))

# 香川県のデータの絞り込み
df_kagawa = df_flow[df_flow['prefecture']=='香川県']
#df['beh_time_date'] = df['beh_time'].dt.strftime('%Y-%m-%d')
df_kagawa['beh_time'] =pd.to_datetime(df_kagawa['beh_time'])
df_kagawa['beh_time_date'] = df_kagawa['beh_time'].dt.strftime('%Y-%m-%d')
current_date = datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.strptime(end_date, '%Y-%m-%d')
df_kagawa_in = df_kagawa[df_kagawa['pas_type']=='pas_in']
df_kagawa_in['arrive_time'] = df_kagawa_in['beh_time'] - pd.to_timedelta(df_kagawa_in['diff_time'], unit='h')
contexts = {}
already_index = set()
while current_date <= end_date:
    print(current_date.strftime('%Y-%m-%d'))
    tommorow_date = current_date+timedelta(days=1)
    
    for k,v in df_kagawa_in[(df_kagawa_in['beh_time_date']==current_date.strftime('%Y-%m-%d')) | (df_kagawa_in['beh_time_date']==tommorow_date.strftime('%Y-%m-%d'))].groupby(['gender', 'age', 'home_prefecture', 'ap21', 'ap22', 'beh_time_date', 'arrive_time']).groups.items():
        #print(k,v)
        v = list(v)
        #if len(df_kagawa_in.loc[v]['mesh'].unique()>3) and len(v)>12:
            #print('new user')
            #print('user profile', k[0], k[1], k[2])
        if True:
            gdf = get_trajectory_gdf(df_kagawa_in.loc[v], kagawa_gdf)
            max_value = df_kagawa_in.loc[v]['diff_time'].value_counts().values.max()
            if max_value>1:
                vs = split_trajectory(v, df_kagawa_in, max_value)
            else:
                vs=[v]
            for i, v in enumerate(vs):
                if v in already_index:continue
                already_index.add(v)
                story = make_trajectory_story(df_kagawa_in.loc[v][['diff_time', 'beh_time_hour', 'mesh']])
                grouped_story, grouped_story_list = aggregate_story(story)
                #print(grouped_story_list)
                context = attach_context(grouped_story_list)
                contexts[(current_date.strftime('%Y-%m-%d'), f'{k[0]}_{k[1]}_{k[2]}_{i}')] = context
                print(context)
                #print(story)
    current_date = tommorow_date
    
    print(contexts)
    with open('../data/context_all.pkl', 'wb') as f:
        pickle.dump(contexts, f)