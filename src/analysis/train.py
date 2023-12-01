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
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from PIL import Image
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
import argparse
from utils import prefecture_region, season
    
def calc_accuracy(model, save_name):
    pred = model.predict(X_test).argmax(1)
    report = classification_report(y_test, pred)

    print("Classification Report:")
    #print(report)

    # 正解率の計算
    accuracy = accuracy_score(y_test, pred)

    print(f"Accuracy: {round(accuracy, 3)}")
    # 適合率、再現率、F1スコアを計算
    precision, recall, f1, support = precision_recall_fscore_support(y_test, pred, average='macro')

    print(f"Macro Precision: {round(precision, 3)}")
    print(f"Macro Recall: {round(recall, 3)}")
    print(f"Macro F1 Score: {round(f1, 3)}")
    plt.figure()
    cm = confusion_matrix(y_pred=pred, y_true=y_test)
    cmp = ConfusionMatrixDisplay(cm, display_labels=['中部', '小豆島', '東部', '西部', '高松'])
    plt.title(f'f1: {round(f1, 3)} Accuracy: {round(accuracy, 3)}')
    cmp.plot(cmap=plt.cm.Blues)
    
    plt.savefig(f'../data/confusion_matrix_optuna_{save_name}.jpg')
    
def save_model(model, save_name):
    with open(f'../data/model/lgbm/lgbm_optuna_{save_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='')
parser.add_argument('--target', type=str, default='region')
parser.add_argument('--over_sample', action='store_true')
parser.add_argument('--under_sample', action='store_true')
args = parser.parse_args()


df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/df_kagawa.csv')
    
X = df_kagawa[df_kagawa['mesh']!=51342061]
X = X[X['home_prefecture']!='香川県']
orig_X = X.copy()
X['home_prefecture_region'] = X['home_prefecture'].map(prefecture_region)
X['beh_time'] = pd.to_datetime(X['beh_time'])
X['beh_time_hour'] = X['beh_time'].dt.hour
X['beh_time_day'] = X['beh_time'].dt.day
X['beh_time_month'] = X['beh_time'].dt.month
X['beh_time_year'] = X['beh_time'].dt.year
X['beh_time_weekday'] = X['beh_time'].dt.weekday
X['beh_time_holiday'] = (X['beh_time_weekday']>=5).astype(int)
X['pas_time'] = pd.to_datetime(X['beh_time'])
X['beh_time_hour'] = X['pas_time'].dt.hour
X['pas_time_day'] = X['pas_time'].dt.day
X['pas_time_month'] = X['pas_time'].dt.month
X['pas_time_weekday'] = X['pas_time'].dt.weekday
X['pas_time_holiday'] = (X['pas_time_weekday']>=5).astype(int)
X['pas_time_hour'] = X['pas_time'].dt.hour
X['season'] = X['pas_time_month'].apply(season)

for col in ['age', 'gender', 'home_prefecture', 'ap21', 'ap22', 'beh_time_year', 'home_prefecture_region', 'pas_type']:
       le = LabelEncoder()
       X[col] = le.fit_transform(X[col].astype(str))
X_ = X[['home_prefecture','home_prefecture_region', 'gender', 'age', 'num',
       'beh_time_hour', 'beh_time_day', 'beh_time_month', 'beh_time_year', 'beh_time_weekday','beh_time_holiday',
       'pas_time_day', 'pas_time_month', 'pas_time_weekday', 'diff_time', 'pas_time_holiday','season',
       'ap21', 'ap22', 'pas_type']]

le = LabelEncoder()
print(X_.shape)
#X['target'] = le.fit_transform(X['mesh'])
X['target'] = le.fit_transform(X[args.target])
y = X['target']

cat_cols = ['home_prefecture','home_prefecture_region', 'gender', 'age',
        'beh_time_day', 'beh_time_month', 'beh_time_year', 'beh_time_weekday','beh_time_holiday',
       'pas_time_day', 'pas_time_month', 'pas_time_weekday', 'pas_time_holiday','season',
       'ap21', 'ap22', 'pas_type']

#for cat in cat_cols:
#    X_[cat] = X_[cat].astype('category')
    
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
d = {'X': {'train': X_train, 'val': X_valid, 'test': X_test},
    'y': {'train': y_train, 'val': y_valid, 'test': y_test},
    'orig_X': orig_X,
    'orig_X_encoded': X}
save_pkl(f'../data/processed_data_{args.save_name}.pkl', d)

over_sample=args.over_sample
under_sample=args.under_sample
if args.under_sample:
    if args.target=='city':under_sample_count=200000
    elif args.target=='region':under_sample_count=200000
    target_count = {k:min(v, under_sample_count) for k,v in zip(y_train.value_counts().index, y_train.value_counts().values)}
    ros = RandomUnderSampler(sampling_strategy=target_count, random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
if args.over_sample:
    if args.target=='city':over_sample_count=50000
    elif args.target=='region':over_sample_count=20000
    target_count = {k:max(v, over_sample_count) for k,v in zip(y_train.value_counts().index, y_train.value_counts().values)}
    #ros = RandomOverSampler(sampling_strategy=target_count, random_state=0)
    cat_cols_num = []
    for i, col in enumerate(X_train.columns):
        if col in cat_cols:
            cat_cols_num.append(i)
    ros = SMOTENC(categorical_features=cat_cols_num)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print('finish resample')
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',  # 多クラス分類
    'num_class': X['target'].max()+1,  # クラスの数
    'metric': 'multi_logloss',  # 評価指標
    'boosting_type': 'gbdt',  # 勾配ブースティング
}

num_round = 1000  # 学習の反復回数
best_params, tuning_history = dict(), list()
bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], 
                categorical_feature=cat_cols,
                callbacks=[early_stopping(50), log_evaluation(50)])

        
#with open('/home/yamanishi/project/airport/src/data/model/best_lgbm_optuna_1000epoch.pkl', 'rb') as f:
#    bst = pickle.load(f)      
print('loaded model')
print(bst.best_score['valid_0']['multi_logloss'])
print(bst.params)
calc_accuracy(bst, args.save_name)
explainer = shap.TreeExplainer(model=bst)
print(explainer.expected_value)
X_test_shap = X_test.copy().reset_index(drop=True)
shap_values = explainer.shap_values(X=X_test_shap)
save_pkl(f'../data/explainer_optuna_{args.exp_name}.pkl', explainer)
save_pkl(f'../data/shap_values_optuna_{args.exp_name}.pkl', shap_values)
save_pkl(f'../data/model/best_lgbm_optuna_{args.exp_name}.pkl', bst)

plt.figure()
shap.summary_plot(shap_values, X_test_shap) #左側の図
plt.savefig(f'../data/shap_summary_optuna_{args.save_name}.jpg')
plt.figure()
shap.summary_plot(shap_values, X_test_shap, plot_type='bar') #右側の図
plt.savefig(f'../data/shap_summary_bar_optuna_{args.save_name}.jpg')

n = 0#テストデータのn番目の要素を指定
shap.force_plot(10, shap_values[n, :])
shap.plots._waterfall.waterfall_legacy(10, 
                                shap_values[n,:], X_test_shap.iloc[n,:])

plt.figure()
n = 0#テストデータのn番目の要素を指定
shap.force_plot(10, shap_values[n, :])
shap.plots._waterfall.waterfall_legacy(10, 
                                shap_values[n,:], X_test_shap.iloc[n,:])
plt.savefig(f'../data/shap_test_optuna_0_{args.save_name}.jpg') 



