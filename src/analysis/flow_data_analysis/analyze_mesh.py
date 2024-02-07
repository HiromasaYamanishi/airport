import numpy
import pandas as pd
import pickle
from typing import List, Dict
import matplotlib.pyplot as plt
import japanize_matplotlib

class MeshAnalyzer:
    def __init__(self,):
        with open('../data/kagawa_gdf.pkl', 'rb') as f:
            self.kagawa_gdf = pickle.load(f)
        
        self.mesh_city = dict(zip(self.kagawa_gdf['mesh'], self.kagawa_gdf['city']))
        self.mesh_region = dict(zip(self.kagawa_gdf['mesh'], self.kagawa_gdf['region']))
        self.df_spot_kagawa = pd.read_csv('../data/df_jalan_kagawa.csv').dropna(subset='review_count')
        self.df_hotel_kagawa = pd.read_csv('../data/hotel/hotel_kagawa.csv').dropna(subset='review_count')
        self.df_food_kagawa = pd.read_csv('../data/food_info_all.csv').dropna(subset='review_count')
        self.df_topic_poi =  pd.concat([self.df_spot_kagawa, self.df_food_kagawa,])
        self.df_kagawa_in = pd.read_csv('../data/df_kagawa_in.csv')
        self.df_kagawa_in['beh_time'] = pd.to_datetime(self.df_kagawa_in['beh_time'])
        
    def plot_time_series(self, df_flow: pd.DataFrame, conditions: List[Dict], plot_all=False, events=[],
                         save_name=''):
        '''
        plot monthly time changes of people flow
        conditions: List of condition dict
        age: ≈', '22～29歳', '60～69歳', '50～59歳', nan, '35～39歳', '18～19歳',
        '30～34歳', '70歳～', '20～21歳', '15～17歳'
        gender: ['男性', '女性', nan]
        '''
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 16
        plt.figure()
        ax = plt.gca()  
        for condition in conditions:
            df_tmp = df_flow.copy()
            for var, cond in condition.items():
                df_tmp = df_tmp[df_tmp[var]==cond]
            df_tmp['count'] = 1
            df_count = df_tmp[['beh_time', 'count']]
            df_count.set_index('beh_time', inplace=True)
            monthly_aggregate = df_count.resample('M').sum()
            plt.xticks(rotation=90)
            plt.plot(monthly_aggregate,label=' '.join([str(v) for v in condition.values()]))
            
        if plot_all:
            df_tmp = df_flow.copy()
            df_tmp['count'] = 1
            df_count = df_tmp[['beh_time', 'count']]
            df_count.set_index('beh_time', inplace=True)
            monthly_aggregate = df_count.resample('M').sum()
            plt.plot(monthly_aggregate,label='all')
            
        for event in events:
            if len(event)==3:
                start_time, end_time, event_name = event
                ax.axvspan(start_time, end_time, color='yellow', alpha=0.3) 
                ax.text(start_time, ax.get_ylim()[1], event_name, rotation=90, verticalalignment='bottom')
            else:
                event_time, event_name = event
                ax.axvline(x=event_time, color='k', linestyle='--')  # Add vertical line for each event
                ax.text(event_time, ax.get_ylim()[1], event_name, rotation=90, verticalalignment='bottom')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'../data/result/{save_name}_ts.png')
        
if __name__ == '__main__':
    mesh_analyzer = MeshAnalyzer()
    mesh_analyzer.plot_time_series(mesh_analyzer.df_kagawa_in,
                                   conditions=[{'group': 'M1',},
                                               {'group': 'M2',},
                                               {'group': 'M3',},
                                               ],
                                   plot_all=False,
                                   save_name='all_male')
    mesh_analyzer.plot_time_series(mesh_analyzer.df_kagawa_in,
                                   conditions=[{'group': 'F1',},
                                               {'group': 'F2',},
                                               {'group': 'F3',},
                                               ],
                                   plot_all=False,
                                   save_name='all_female')
    exit()
    mesh_analyzer.plot_time_series(mesh_analyzer.df_kagawa_in,
                                   conditions=[{'group': 'M1', 'city': '小豆郡土庄町'},
                                               {'group': 'M2', 'city': '小豆郡土庄町'},
                                               {'group': 'M3', 'city': '小豆郡土庄町'},
                                               ],
                                   plot_all=False,
                                   events=[(pd.Timestamp('2019-07-01'), pd.Timestamp('2019-09-01'),'アニメ2期'),
                                           (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-01'),'アニメ3期'),
                                           (pd.Timestamp('2022-06-01'), pd.Timestamp('2022-08-01'), '劇場化'),],
                                   save_name='syodoshima_male')
    mesh_analyzer.plot_time_series(mesh_analyzer.df_kagawa_in,
                                   conditions=[{'group': 'F1', 'city': '小豆郡土庄町'},
                                               {'group': 'F2', 'city': '小豆郡土庄町'},
                                               {'group': 'F3', 'city': '小豆郡土庄町'},
                                               ],
                                    events=[(pd.Timestamp('2019-07-01'), pd.Timestamp('2019-09-01'),'アニメ2期'),
                                           (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-01'),'アニメ3期'),
                                           (pd.Timestamp('2022-06-01'), pd.Timestamp('2022-08-01'), '劇場化'),],
                                   plot_all=False,
                                   save_name='syodoshima_female')