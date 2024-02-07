import MeCab
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter,TokenCountFilter, CompoundNounFilter
from janome.charfilter import RegexReplaceCharFilter, UnicodeNormalizeCharFilter
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from typing import List
import re

def join_nouns(text):
    if pd.isna(text):
        return None
    text = text.replace('<|im_end|>', '')
    char_filters = [
        UnicodeNormalizeCharFilter(),
        RegexReplaceCharFilter('[#!:;<>{}・`.,()-=$/_\d\'"\[\]\|年月日~]+', ' '),
    ]
    token_filters = [
        POSKeepFilter(['名詞']),
        #CompoundNounFilter()
        #TokenCountFilter(),
    ]
    analyzer = Analyzer(char_filters = char_filters, token_filters = token_filters)
    token_nouns = analyzer.analyze(text)
    token_nouns = [l.surface for l in token_nouns]
    joint_nouns = (' ').join(token_nouns)
    return joint_nouns

def tokenize_and_join_text(texts: List[str]):
    texts = [join_nouns(text) for text in tqdm(texts)]
    texts = ' '.join(texts)
    return texts

def make_and_save_wordcloud(texts, save_name):
    words = ["私","わたし","僕","あなた","みんな","ただ","ほか","それ", "もの", "これ", "ところ","ため","うち","ここ",
                "そう","どこ", "つもり", "いつ","あと","もん","はず","こと","そこ","あれ",
                "なに","傍点","まま","事","人","方","何","時","一","二","三","四","五","六","七","八","九","十",
                'の', 'よう', 'さ', 'さん', 'たち', '多数派', '短文', '集合', '説明', '要約', 'トピック', '多数', '派', '該当']
    result = WordCloud(width=800, height=600, background_color='white', font_path='/home/yamanishi/project/airport/src/data/Noto_Sans_JP/NotoSansJP-VariableFont_wght.ttf', regexp=r"[\w']+", stopwords=words).generate(texts)
    plt.figure(figsize=(12,10))
    plt.imshow(result)
    plt.savefig(f'../data/wordcloud/wordcloud_{save_name}.png')
    
def make_island_review():
    df_review = pd.read_csv('../data/kagawa_review.csv')
    df_spot = pd.read_csv('../data/df_jalan_kagawa.csv')
    df1 = df_spot[df_spot['city'].isin(['小豆島町（小豆郡）', '土庄町（小豆郡）'])]
    #df1 = df_spot[df_spot['city'].isin(['直島町（香川郡）', '小豆島町（小豆郡）', '土庄町（小豆郡）'])]
    df2 = df_spot[df_spot['spot_name'].str.contains('男木')]
    df3 = df_spot[df_spot['spot_name'].str.contains('佐柳')]
    df4 = df_spot[df_spot['spot_name'].str.contains('女木')]
    df5 = df_spot[df_spot['spot_name'].str.contains('与島')]
    df_island = pd.concat([df5, df2, df3, df4])
    #df_island = df1
    df_review_island = df_review[df_review['spot'].isin(df_island['spot_name'])]
    return df_review_island
       
       
        
def prepare_data_posneg():
    with open('../data/review/goodbad_all_incontext_kagawa_0.pkl', 'rb') as f:
        goodbad0 = pickle.load(f)
    with open('../data/review/goodbad_all_incontext_kagawa_1.pkl', 'rb') as f:
        goodbad1 = pickle.load(f)
    with open('../data/review/goodbad_all_incontext_kagawa_2.pkl', 'rb') as f:
        goodbad2 = pickle.load(f)
    goodbad0.update(goodbad1)
    goodbad0.update(goodbad2)
    
    df_review_island = make_island_review()
    goodbad0 = {ind:goodbad0[ind] for ind in list(df_review_island.index)}
    
    pos_sentence, neg_sentence = [], []
    for i,(k,v) in enumerate(goodbad0.items()):
        matches = re.findall(r'「([^」]+)」', v)
        negative_ind = v.find('ネガティブな')
        posneg = [(v.find(m)<negative_ind) for m in matches]
        for m, pn in zip(matches, posneg):
            if pn:pos_sentence.append(m)
            else:neg_sentence.append(m)
    return pos_sentence, neg_sentence

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, )
    parser.add_argument('--save_suffix', type=str, default='')
    parser.add_argument('--posneg', type=str, default='')
    parser.add_argument('--gender', type=str, default='')
    parser.add_argument('--island', action='store_true')
    args = parser.parse_args()
    if not args.island:
        if 'topic' in args.df_path:
            df = pd.read_csv(args.df_path, names=['index', 'spot', 'posneg', 'cluster', 'topic'])
            target_col = 'topic'
        else:
            df = pd.read_csv(args.df_path)#'/home/yamanishi/project/airport/src/data/kagawa_review.csv')
            target_col = 'review'
        
        if args.posneg=='pos':
            df = df[df['posneg']=='pos']
        elif args.posneg=='neg':
            df = df[df['posneg']=='neg']
            
        if args.gender=='male':
            df = df[df['sex']=='男性']
        elif args.gender=='female':
            df = df[df['sex']=='女性']
            
        print('sample num', len(df))
        texts = list(df[target_col].values)
        texts = tokenize_and_join_text(texts)
        words = ["私","わたし","僕","あなた","みんな","ただ","ほか","それ", "もの", "これ", "ところ","ため","うち","ここ",
                "そう","どこ", "つもり", "いつ","あと","もん","はず","こと","そこ","あれ",
                "なに","傍点","まま","事","人","方","何","時","一","二","三","四","五","六","七","八","九","十",
                'の', 'よう', 'さ', 'さん', 'たち', '多数派', '短文', '集合', '説明', '要約', 'トピック', '多数', '派', '該当']
        result = WordCloud(width=800, height=600, background_color='white', font_path='/home/yamanishi/project/airport/src/data/Noto_Sans_JP/NotoSansJP-VariableFont_wght.ttf', regexp=r"[\w']+", stopwords=words).generate(texts)
        plt.figure(figsize=(12,10))
        plt.imshow(result)
        plt.savefig(f'../data/wordcloud/wordcloud_{args.save_suffix}.png')
    else:
        pos_sentences, neg_sentences = prepare_data_posneg()
        pos_sentences = tokenize_and_join_text(pos_sentences)
        neg_sentences = tokenize_and_join_text(neg_sentences)
        make_and_save_wordcloud(pos_sentences, 'other_island_pos')
        make_and_save_wordcloud(neg_sentences, 'other_island_neg')

        