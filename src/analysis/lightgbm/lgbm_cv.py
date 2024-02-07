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
from matplotlib.animation import PillowWriter
import matplotlib
import functools
import shap
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.animation as animation
from lightgbm import early_stopping
from lightgbm import log_evaluation
import optuna.integration.lightgbm as lgb
from PIL import Image

def season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    else:
        return 3
    
df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/df_kagawa.csv')
    
X = df_kagawa[df_kagawa['mesh']!=51342061]
X = X[X['home_prefecture']!='香川県']

X['beh_time'] = pd.to_datetime(X['beh_time'])
X['beh_time_hour'] = X['beh_time'].dt.hour
X['beh_time_day'] = X['beh_time'].dt.day
X['beh_time_month'] = X['beh_time'].dt.month
X['beh_time_weekday'] = X['beh_time'].dt.weekday
X['beh_time_holiday'] = (X['beh_time_weekday']>=5).astype(int)
X['pas_time'] = pd.to_datetime(X['beh_time'])
X['beh_time_hour'] = X['pas_time'].dt.hour
X['pas_time_day'] = X['pas_time'].dt.day
X['pas_time_month'] = X['pas_time'].dt.month
X['pas_time_weekday'] = X['pas_time'].dt.weekday
X['pas_time_holiday'] = (X['pas_time_weekday']>=5).astype(int)
X['season'] = X['pas_time_month'].apply(season)

for col in ['age', 'gender', 'home_prefecture']:
       le = LabelEncoder()
       X[col] = le.fit_transform(X[col].astype(str))
X_ = X[['home_prefecture', 'gender', 'age', 'num',
       'beh_time_hour', 'beh_time_day', 'beh_time_month', 'beh_time_weekday','beh_time_holiday',
       'pas_time_day', 'pas_time_month', 'pas_time_weekday', 'diff_time', 'pas_time_holiday','season']]

le = LabelEncoder()
#X['target'] = le.fit_transform(X['mesh'])
X['target'] = le.fit_transform(X['region'])
y = X['target']

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',  # 多クラス分類
    'num_class': X['target'].max()+1,  # クラスの数
    'metric': 'multi_logloss',  # 評価指標
    'boosting_type': 'gbdt',  # 勾配ブースティング
}

num_round = 1000 # 学習の反復回数
early_stop_count = 50
best_params, tuning_history = dict(), list()
bst = lgb.train(params, 
                train_data, 
                num_round, 
                valid_sets=[test_data], 
                callbacks=[early_stopping(early_stop_count), 
                           log_evaluation(early_stop_count)])

print(best_params, tuning_history)
with open('../data/model/best_lgbm_optuna_2000epoch.pkl', 'wb') as f:
    pickle.dump(bst, f)
with open('../data/tuning_history.pkl', 'wb') as f:
    pickle.dump(tuning_history)
with open('../data/best_params.pkl', 'wb') as f:
    pickle.dump(bst.best_params_)