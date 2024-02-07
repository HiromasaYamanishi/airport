import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import pickle


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

def static_test(df, gdf, condition, compare_condition, thresh=0.2):
    target_df = df.copy()
    for k,v in condition.items():
        target_df = target_df[target_df[k]==v]
    target_gdf = get_count_gdf(target_df, gdf)
    target_index = target_df.index
    
    compare_df = df.copy()    
    for k,v in compare_condition.items():
        compare_df = compare_df[compare_df[k]==v]
    compare_df = compare_df[~compare_df.index.isin(target_index)]
    compare_gdf = get_count_gdf(compare_df, gdf)
    
    assert len(target_gdf)==len(compare_gdf)
    
    target_total = sum(target_gdf['count'])
    compare_total = sum(compare_gdf['count'])
    pvalues = []
    for mesh, target_count, compare_count in zip(target_gdf['mesh'], target_gdf['count'], compare_gdf['count']):
        #print(target_count, target_total, compare_count, compare_total)
        pvalue = proportions_ztest([target_count, target_total], [compare_count, compare_total],alternative='larger')[1]
        pvalues.append((mesh, pvalue))
    pvalues.sort(key=lambda x:x[1])
    poi_all = []
    for mesh, pvalue in pvalues:
        #print(pvalue)
        if not pd.isna(pvalue) and pvalue<thresh:
            #print(mesh)
            spots = df_jalan_kagawa[df_jalan_kagawa['mesh']==mesh][['spot_name', 'review_count']].values
            hotels = df_hotel_kagawa[df_hotel_kagawa['mesh']==mesh][['hotel_name', 'review_count']].values
            foods = df_jalan_food[df_jalan_food['mesh']==mesh][['food_name', 'review_count']].values
            poi_all.append((mesh, spots, hotels, foods))
    return poi_all

if __name__=='__main__':
    df_jalan_kagawa = pd.read_csv('../data/df_jalan_kagawa.csv')
    df_jalan_food = pd.read_csv('../data/df_jalan_food.csv')
    df_hotel_kagawa = pd.read_csv('../data/df_hotel_kagawa.csv')
    kagawa_gdf = pd.read_csv('../data/kagawa_gdf.csv')
    df_kagawa_in = pd.read_csv('../data/df_kagawa_in.csv')
    
    poi_alls = {}
    thresh = 0.01
    for gender in ['男性', '女性']:
        for age_group in ['young', 'middle', 'senior']:
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age_group': age_group, 'gender': gender}, {}, thresh=thresh)
            poi_alls[(gender+age_group, '')] = poi_all
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age_group': age_group, 'gender': gender}, {'gender': gender}, thresh=thresh)
            poi_alls[(gender+age_group, gender)] = poi_all
    with open('../data/static_test/test_result_group_rc_001.pkl', 'wb') as f:
        pickle.dump(poi_alls, f)
        
    poi_alls = {}
    thresh = 0.05
    for gender in ['男性', '女性']:
        for age_group in ['young', 'middle', 'senior']:
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age_group': age_group, 'gender': gender}, {}, thresh=thresh)
            poi_alls[(gender+age_group, '')] = poi_all
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age_group': age_group, 'gender': gender}, {'gender': gender}, thresh=thresh)
            poi_alls[(gender+age_group, gender)] = poi_all
    with open('../data/static_test/test_result_group_rc_005.pkl', 'wb') as f:
        pickle.dump(poi_alls, f)
    exit()
            
    poi_alls = {}
    thresh=0.01
    for gender in ['男性', '女性']:
        poi_all= static_test(df_kagawa_in, kagawa_gdf, {'gender': gender}, {}, thresh=0.01)
        poi_alls[(gender, '')] = poi_all
        
    for age in ['50～59歳','35～39歳', '22～29歳', '70歳～', '60～69歳', '18～19歳',
       '30～34歳', '40～49歳', '20～21歳', '15～17歳']:
        poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age': age}, {}, thresh=0.01)
        poi_alls[(age, '')] = poi_all
        
    for gender in ['男性', '女性']:
        for age in ['50～59歳','35～39歳', '22～29歳', '70歳～', '60～69歳', '18～19歳',
        '30～34歳', '40～49歳', '20～21歳', '15～17歳']:
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age': age, 'gender': gender}, {}, thresh=thresh)
            poi_alls[(gender+age, '')] = poi_all
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age': age, 'gender': gender}, {'gender': gender}, thresh=thresh)
            poi_alls[(gender+age, gender)] = poi_all
            
    with open('../data/static_test/test_result_001.pkl', 'wb') as f:
        pickle.dump(poi_alls, f)
        
    poi_alls = {}
    thresh=0.05      
    for gender in ['男性', '女性']:
        poi_all= static_test(df_kagawa_in, kagawa_gdf, {'gender': gender}, {}, thresh=thresh)
        poi_alls[(gender, '')] = poi_all
        
    for age in ['50～59歳','35～39歳', '22～29歳', '70歳～', '60～69歳', '18～19歳',
       '30～34歳', '40～49歳', '20～21歳', '15～17歳']:
        poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age': age}, {}, thresh=thresh)
        poi_alls[(age, '')] = poi_all
        
    for gender in ['男性', '女性']:
        for age in ['50～59歳','35～39歳', '22～29歳', '70歳～', '60～69歳', '18～19歳',
        '30～34歳', '40～49歳', '20～21歳', '15～17歳']:
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age': age, 'gender': gender}, {}, thresh=thresh)
            poi_alls[(gender+age, '')] = poi_all
            poi_all= static_test(df_kagawa_in, kagawa_gdf, {'age': age, 'gender': gender}, {'gender': gender}, thresh=thresh)
            poi_alls[(gender+age, gender)] = poi_all
    with open('../data/static_test/test_result_005.pkl', 'wb') as f:
        pickle.dump(poi_alls, f)