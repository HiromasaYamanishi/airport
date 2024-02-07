import pickle
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.clustering import silhouette_score
from tqdm import tqdm
import re
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Point
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import folium
from tslearn.utils import to_time_series_dataset
from tslearn.barycenters import dtw_barycenter_averaging
from folium import GeoJson
from collections import defaultdict

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
def get_color(review_count):
    if review_count>500:
        return 'red'
    elif review_count>100:
        return 'orange'
    elif review_count>50:
        return 'beige'
    elif review_count>10:
        return 'green'
    else:
        return 'blue'
def get_color_food(review_count):
    if review_count>300:
        return 'red'
    elif review_count>60:
        return 'orange'
    elif review_count>30:
        return 'beige'
    elif review_count>10:
        return 'green'
    
def style_function_ratio(feature):
    value = feature['properties']['count']  # メッシュの値を取得
    if value>0.05:
        color='red'
    elif value>0.01:
        color='orange',
    elif value>0.002:
        color='yellow'
    elif value>0.0004:
        color='green'
    else:
        color='blue'# 値に応じて色を設定
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    }
    
def style_function_ratio_around(feature):
    value = feature['properties']['count']  # メッシュの値を取得
    if value>0.05:
        color='red'
    elif value>0.005:
        color='orange',
    elif value>0.0005:
        color='yellow'
    elif value>0.00005:
        color='green'
    else:
        color='blue'# 値に応じて色を設定
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    }
    
def style_function(feature):
    value = feature['properties']['count']  # メッシュの値を取得
    if value>100000:
        color='red'
    elif value>10000:
        color='orange',
    elif value>1000:
        color='yellow'
    elif value>100:
        color='green'
    else:
        color='blue'# 値に応じて色を設定
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    }

def style_function_time(feature):
    value = feature['properties']['count']  # メッシュの値を取得
    if value>10000:
        color='red'
    elif value>1000:
        color='orange',
    elif value>100:
        color='yellow'
    elif value>10:
        color='green'
    else:
        color='blue'# 値に応じて色を設定
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    }
def get_count_gdf(df_flow, kagawa_gdf):
    #print(df_flow['mesh'])
    mesh_counts = df_flow['mesh'].value_counts()
    mesh_visit_count_df = mesh_counts.reset_index()
    mesh_visit_count_df.columns = ['mesh', 'count']
    gdf_tmp = kagawa_gdf.copy()
    if 'count' in gdf_tmp.columns:
        gdf_tmp = gdf_tmp.drop('count', axis=1)
        gdf_tmp['mesh'] = gdf_tmp['code'].astype(int)
    #print(mesh_visit_count_df, gdf_tmp)
    gdf_tmp = gdf_tmp.merge(mesh_visit_count_df, on='mesh', how='left')
    gdf_tmp['count']=gdf_tmp['count'].fillna(0)
    return gdf_tmp
    

with open("../data/shp/37kagawa1km.geojson") as f:
    kagawa = json.load(f)
kagawa_gdf = gpd.GeoDataFrame.from_features(kagawa['features'])
kagawa_gdf['mesh'] = kagawa_gdf['code'].astype(int)
df_hotel = pd.read_csv('/home/yamanishi/project/airport/src/data/hotel_info.csv')
df_jalan = pd.read_csv('../data/experience_light.csv')
df_jalan_food = pd.read_csv('/home/yamanishi/project/airport/src/data/food_info_all.csv')
df_hotel_kagawa = df_hotel[df_hotel['prefecture']=='香川県']
df_jalan_kagawa = df_jalan[df_jalan['prefecture']=='香川県']
df_jalan_food = get_mesh_spot(df_jalan_food, kagawa_gdf)
df_jalan_kagawa = get_mesh_spot(df_jalan_kagawa, kagawa_gdf)
df_hotel_kagawa = get_mesh_spot(df_hotel_kagawa, kagawa_gdf)
df_kagawa_in = pd.read_csv('/home/yamanishi/project/airport/src/data/df_kagawa_in.csv')

def make_kagawa_map(gdf, style_function, ratio=False, add_icon=False, add_icon_food=False,
                    add_icon_hotel=False,df_jalan=df_jalan_kagawa, df_jalan_food=df_jalan_food, 
                    df_hotel=df_hotel_kagawa, add_color_bar=None, icon_size=6):
    m = folium.Map(location=[34.3, 134], zoom_start=11)
    if ratio:
        gdf['abs_count'] = gdf['count']
        gdf['count'] = gdf['count']/sum(gdf['count'])
    mesh_json = gdf.to_json()
    GeoJson(mesh_json, style_function=style_function,name='Mesh Layer').add_to(m)
    if add_icon:
        for ind, spot_name, lat,lon, review_count in zip(df_jalan.index, df_jalan['spot_name'], df_jalan['latitude'], df_jalan['longitude'], df_jalan['review_count']):
        # マーカーにpopupを追加
        #iframe = folium.IFrame('''<div style="font-size: 20px;"> {} </div>'''.format(spot_name), )
            if review_count<10:continue
            #print(spot_name)
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(spot_name, parse_html=True),
                icon=folium.Icon(color=get_color(review_count), icon_size=icon_size)
            ).add_to(m)
    
    if add_icon_food:
        for ind, spot_name, lat,lon, review_count in zip(df_jalan.index, df_jalan_food['food_name'], df_jalan_food['latitude'], df_jalan_food['longitude'], df_jalan_food['review_count']):
        # マーカーにpopupを追加
        #iframe = folium.IFrame('''<div style="font-size: 20px;"> {} </div>'''.format(spot_name), )
            if review_count<20:continue
            if pd.isna(lat) or pd.isna(lon):continue
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(spot_name, parse_html=True),
                icon=folium.Icon(color=get_color_food(review_count), icon='cutlery', icon_size=icon_size)
            ).add_to(m)
            
    if add_icon_hotel:
        for ind, spot_name, lat,lon, review_count in zip(df_hotel.index, df_hotel['hotel_name'], df_hotel['latitude'], df_hotel['longitude'], df_hotel['review_count']):
        # マーカーにpopupを追加
        #iframe = folium.IFrame('''<div style="font-size: 20px;"> {} </div>'''.format(spot_name), )
            if review_count<5:continue
            if pd.isna(lat) or pd.isna(lon):continue
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(spot_name, parse_html=True),
                icon=folium.Icon(color=get_color(review_count), icon='bed', icon_size=icon_size)
            ).add_to(m)
        
        
    return m

color_names = """khaki, maroon, aqua, aquamarine, lightgrey,
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
            beige, aliceblue, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, azure, lightgrey,
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

def load_trajectory(region='naoshima', segment=''):
    with open(f'../data/trajectory/trajectory{segment}_{region}.pkl', 'rb') as f:
        trajectory = pickle.load(f)
        
    return trajectory


def calc_silhouette(data, method, n_clusters, metric):
    # metric: [dtw, softdtw, euclidian]
    if method=='kmeans':
        dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
                                n_init=5,
                                metric=metric,
                                verbose=False,
                                random_state=22)
        
    elif method=='kshape':
        dba_km = KernelKMeans(n_clusters=n_clusters, kernel="gak", random_state=0)
    label = dba_km.fit_predict(data)
    score = silhouette_score(data, label)
    return score


if __name__ == '__main__':
    with open('../data/trajectory/trajectory_shodoshima.pkl', 'rb') as f:
        mesh_all_syodoshima_d = pickle.load(f)
    with open('../data/trajectory/trajectory_female_shodoshima.pkl', 'rb') as f:
        mesh_female_syodoshima_d  = pickle.load(f)
    with open('../data/trajectory/trajectory_female_young_shodoshima.pkl', 'rb') as f:
        mesh_female_young_syodoshima_d  = pickle.load(f)
    with open('../data/trajectory/trajectory_male_shodoshima.pkl', 'rb') as f:
        mesh_male_syodoshima_d = pickle.load(f)
        
    with open('../data/trajectory/trajectory_naoshima.pkl', 'rb') as f:
        mesh_all_naoshima_d = pickle.load(f)
    with open('../data/trajectory/trajectory_female_naoshima.pkl', 'rb') as f:
        mesh_female_naoshima_d = pickle.load(f)
    with open('../data/trajectory/trajectory_female_young_naoshima.pkl', 'rb') as f:
        mesh_female_young_naoshima_d = pickle.load(f)
    with open('../data/trajectory/trajectory_male_naoshima.pkl', 'rb') as f:
        mesh_male_naoshima_d = pickle.load(f)
        
    with open('../data/trajectory/trajectory_all.pkl', 'rb') as f:
        mesh_all_d = pickle.load(f)
    with open('../data/trajectory/trajectory_female_all.pkl', 'rb') as f:
        mesh_female_d = pickle.load(f)
    with open('../data/trajectory/trajectory_female_young_all.pkl', 'rb') as f:
        mesh_female_young_d = pickle.load(f)
    with open('../data/trajectory/trajectory_male_all.pkl', 'rb') as f:
        mesh_male_d = pickle.load(f)
        
    # dba_km = TimeSeriesKMeans(n_clusters=3,
    #                         n_init=5,
    #                         metric="dtw",
    #                         verbose=True,
    #                         random_state=22)
    # label = dba_km.fit_predict(mesh_all_syodoshima_d)
    
    # gdf = get_count_gdf(df_kagawa_in, kagawa_gdf)
    # for i in range(3):
    #     m = make_kagawa_map(gdf, style_function, add_icon=True,df_jalan=df_jalan_kagawa, add_icon_food=True, add_icon_hotel=True)
    #     trajs = []
    #     for traj, c in zip(mesh_all_syodoshima_d, label):
    #         if c==i:
    #             folium.PolyLine(
    #                 locations=traj[~np.isnan(traj).any(axis=1)],     
    #                 color=color_list[c],
    #                 ).add_to(m)
    #             trajs.append(traj)
                
    #     trajectory_center = dtw_barycenter_averaging(trajs)
    #     folium.PolyLine(
    #             locations=trajectory_center,     
    #             color='black',
    #             ).add_to(m)

    #     m.save(f'../data/html/trajectory_cluster/shodoshima_cluster{i}_cnum3.html')
        
    dba_km = TimeSeriesKMeans(n_clusters=3,
                            n_init=5,
                            metric="dtw",
                            verbose=True,
                            random_state=22)
    label = dba_km.fit_predict(mesh_female_young_syodoshima_d)
    u, counts = np.unique(label, return_counts=True)
    print(u, counts)
    score = silhouette_score(mesh_female_young_syodoshima_d, label)
    print(score)
    gdf = get_count_gdf(df_kagawa_in, kagawa_gdf)
    for i in range(3):
        m = make_kagawa_map(gdf, style_function, add_icon=True,df_jalan=df_jalan_kagawa, add_icon_food=True, add_icon_hotel=True)
        trajs = []
        for traj, c in zip(mesh_female_young_syodoshima_d, label):
            if c==i:
                folium.PolyLine(
                    locations=traj[~np.isnan(traj).any(axis=1)],     
                    color=color_list[c],
                    ).add_to(m)
                trajs.append(traj)
                
        trajectory_center = dtw_barycenter_averaging(trajs)
        folium.PolyLine(
                locations=trajectory_center,     
                color='black',
                ).add_to(m)

        m.save(f'../data/html/trajectory_cluster/shodoshima_cluster{i}_female_young_cnum3.html')
        
    # dba_km = TimeSeriesKMeans(n_clusters=3,
    #                         n_init=5,
    #                         metric="dtw",
    #                         verbose=True,
    #                         random_state=22)
    # label = dba_km.fit_predict(mesh_all_naoshima_d)
    # u, counts = np.unique(label, return_counts=True)
    # print(u, counts)
    # score = silhouette_score(mesh_all_naoshima_d, label)
    # print(score)
    # gdf = get_count_gdf(df_kagawa_in, kagawa_gdf)
    # for i in range(3):
    #     m = make_kagawa_map(gdf, style_function, add_icon=True,df_jalan=df_jalan_kagawa, add_icon_food=True, add_icon_hotel=True)
    #     trajs = []
    #     for traj, c in zip(mesh_all_naoshima_d, label):
    #         if c==i:
    #             folium.PolyLine(
    #                 locations=traj[~np.isnan(traj).any(axis=1)],     
    #                 color=color_list[c],
    #                 ).add_to(m)
    #             trajs.append(traj)
                
    #     trajectory_center = dtw_barycenter_averaging(trajs)
    #     folium.PolyLine(
    #             locations=trajectory_center,     
    #             color='black',
    #             ).add_to(m)

    #     m.save(f'../data/html/trajectory_cluster/naoshima_cluster{i}_cnum3.html')
        
    dba_km = TimeSeriesKMeans(n_clusters=3,
                            n_init=5,
                            metric="dtw",
                            verbose=True,
                            random_state=22)
    label = dba_km.fit_predict(mesh_female_naoshima_d) 
    u, counts = np.unique(label, return_counts=True)
    print(u, counts)     
    gdf = get_count_gdf(df_kagawa_in, kagawa_gdf)
    for i in range(3):
        m = make_kagawa_map(gdf, style_function, add_icon=True,df_jalan=df_jalan_kagawa, add_icon_food=True, add_icon_hotel=True)
        trajs = []
        for traj, c in zip(mesh_female_naoshima_d, label):
            if c==i:
                folium.PolyLine(
                    locations=traj[~np.isnan(traj).any(axis=1)],     
                    color=color_list[c],
                    ).add_to(m)
                trajs.append(traj)
                
        trajectory_center = dtw_barycenter_averaging(trajs)
        folium.PolyLine(
                locations=trajectory_center,     
                color='black',
                ).add_to(m)

        m.save(f'../data/html/trajectory_cluster/naoshima_cluster{i}_female_cnum3.html')
        
    # dba_km = TimeSeriesKMeans(n_clusters=3,
    #                         n_init=5,
    #                         metric="dtw",
    #                         verbose=True,
    #                         random_state=22)
    # label = dba_km.fit_predict(mesh_all_d) 
    # gdf = get_count_gdf(df_kagawa_in, kagawa_gdf)
    # for i in range(3):
    #     m = make_kagawa_map(gdf, style_function, add_icon=True,df_jalan=df_jalan_kagawa, add_icon_food=True, add_icon_hotel=True)
    #     trajs = []
    #     for traj, c in zip(mesh_all_d, label):
    #         if c==i:
    #             folium.PolyLine(
    #                 locations=traj[~np.isnan(traj).any(axis=1)],     
    #                 color=color_list[c],
    #                 ).add_to(m)
    #             if not np.isnan(traj).all() or len(traj)>0:
    #                 trajs.append(traj)
                
    #     #trajectory_center = dtw_barycenter_averaging(trajs)
    #     # folium.PolyLine(
    #     #         locations=trajectory_center,     
    #     #         color='black',
    #     #         ).add_to(m)

    #     m.save(f'../data/html/trajectory_cluster/all_cluster{i}_cnum3.html')
    
    for cluster_num in [3,]:
        dba_km = TimeSeriesKMeans(n_clusters=cluster_num,
                                n_init=5,
                                metric="dtw",
                                verbose=True,
                                random_state=22)
        label = dba_km.fit_predict(mesh_female_young_d) 
        u, counts = np.unique(label, return_counts=True)
        print(u, counts)
        gdf = get_count_gdf(df_kagawa_in, kagawa_gdf)
        for i in range(cluster_num):
            m = make_kagawa_map(gdf, style_function, add_icon=True,df_jalan=df_jalan_kagawa, add_icon_food=True, add_icon_hotel=True)
            trajs = []
            for traj, c in zip(mesh_female_young_d, label):
                if c==i:
                    folium.PolyLine(
                        locations=traj[~np.isnan(traj).any(axis=1)],     
                        color=color_list[c],
                        ).add_to(m)
                    if not np.isnan(traj).all() or len(traj)>0:
                        trajs.append(traj)

            #trajectory_center = dtw_barycenter_averaging(trajs)
            # folium.PolyLine(
            #         locations=trajectory_center,     
            #         color='black',
            #         ).add_to(m)

            m.save(f'../data/html/trajectory_cluster/all_cluster{i}_female_young_cnum{cluster_num}.html')
        score = silhouette_score(mesh_female_young_d, label)
        print(cluster_num, score)
    exit()
    for method in ['kmeans', 'kshape']:
        for metric in ['dtw', 'softdtw']:
            for region in ['shodoshima', 'naoshima']:
                for n_cluster in [2, 3, 4,5, 7, 9, 11, 13, 15]:
                    data = load_trajectory(region, '_female')
                    #data = data[:20]
                    score = calc_silhouette(data, method, n_cluster, 'dtw')
                    methods.append(method)
                    n_clusters.append(n_cluster)
                    metrics.append('dtw')
                    scores.append(score)
                    
                    df = pd.DataFrame({'region': [region],
                                    'segment': ['female'],
                                    'method': [method],
                                    'n_cluster': [n_cluster],
                                    'metcis': [metric],
                                    'score': [score]
                                    })
                    df.to_csv('../data/trajectory/clustering_result.csv', mode='a', header=False, index=False)
                
            
    # df = pd.DataFrame({'method': methods,
    #                    'n_cluster': n_clusters,
    #                    'metric': metrics,
    #                    'score': scores})
    
    # df.to_csv('../data/trajectory/clustering_result.csv', mode='a', header=False, index=False)
    
            