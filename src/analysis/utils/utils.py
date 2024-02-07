from folium import GeoJson
import folium
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
prefecture_region = {
    '北海道': '北海道地方',
    '青森県': '東北地方',
    '岩手県': '東北地方',
    '宮城県': '東北地方',
    '秋田県': '東北地方',
    '山形県': '東北地方',
    '福島県': '東北地方',
    '茨城県': '関東地方',
    '栃木県': '関東地方',
    '群馬県': '関東地方',
    '埼玉県': '関東地方',
    '千葉県': '関東地方',
    '東京都': '関東地方',
    '神奈川県': '関東地方',
    '新潟県': '中部地方',
    '富山県': '中部地方',
    '石川県': '中部地方',
    '福井県': '中部地方',
    '山梨県': '中部地方',
    '長野県': '中部地方',
    '岐阜県': '中部地方',
    '静岡県': '中部地方',
    '愛知県': '中部地方',
    '三重県': '近畿地方',
    '滋賀県': '近畿地方',
    '京都府': '近畿地方',
    '大阪府': '近畿地方',
    '兵庫県': '近畿地方',
    '奈良県': '近畿地方',
    '和歌山県': '近畿地方',
    '鳥取県': '中国地方',
    '島根県': '中国地方',
    '岡山県': '中国地方',
    '広島県': '中国地方',
    '山口県': '中国地方',
    '徳島県': '四国地方',
    '香川県': '四国地方',
    '愛媛県': '四国地方',
    '高知県': '四国地方',
    '福岡県': '九州地方',
    '佐賀県': '九州地方',
    '長崎県': '九州地方',
    '熊本県': '九州地方',
    '大分県': '九州地方',
    '宮崎県': '九州地方',
    '鹿児島県': '九州地方',
    '沖縄県': '沖縄地方'
}

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

def season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    else:
        return 3

def get_color(index, cmap_name='gist_rainbow', vmin=0, vmax=1):
    """
    連続的に値を得られるカラーマップから色を計算する関数。
    
    Parameters:
    - index: 色を計算するためのインデックス値。
    - cmap_name: 使用するmatplotlibのカラーマップの名前。
    - vmin, vmax: カラーマップの最小値と最大値。インデックスはこの範囲に正規化される。
    
    Returns:
    - color: 指定されたインデックスに対応する色（16進数形式）。
    """
    # カラーマップのインスタンスを取得
    cmap = plt.get_cmap(cmap_name)
    
    # インデックスをカラーマップの範囲に正規化
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # カラーマップから色を取得し、RGB形式で返す
    rgb = cmap(norm(index))[:3]  # RGBAの最初の3要素を取得
    
    # RGBを16進数形式に変換
    color = mcolors.rgb2hex(rgb)
    
    return color

def get_color_spot(review_count):
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

def make_kagawa_map(gdf, style_function, df_jalan, df_jalan_food,df_hotel, ratio=False, add_icon=False, add_icon_food=False,
                    add_icon_hotel=False,add_color_bar=None, icon_size=12):
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
                icon=folium.Icon(color=get_color_spot(review_count), icon_size=icon_size)
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
                icon=folium.Icon(color=get_color_spot(review_count), icon='bed', icon_size=icon_size)
            ).add_to(m)
        
        
    return m

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
