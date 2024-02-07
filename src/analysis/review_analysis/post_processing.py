import pickle
import numpy as np
from vllm import LLM, SamplingParams
import argparse
from collections import defaultdict
from functools import partial
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def make_prompt(topics, llm):
    messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
    
    prompt = '入力として箇条書きの複数のトピックを与えます。同じトピックがあれば統合して一つのトピックにまとめてください。そして, 独立なトピックを箇条書きで出力してください.次は例です.\n\
        例1:\n\
        入力:\n\
            ・暑い夏の夕方で、人も多かった\n\
            ・一日に二回道が現れては消える\n\
            ・真夏の暑さ\n\
            ・道が細い\n\
            ・干潮時間に合わせて行かないと渡れない\n\
            ・干潮や満潮により道が出てきたりなくなったりする\n\
            ・GWや時間によっては混雑する\n\
        出力:\n\
            ・真夏の暑さ\n\
            ・道が細い\n\
            ・干潮と満潮の1日2回しか道が渡れない\n\
            ・GWや時間によっては混雑する\n\
        \n\
        例2:\n\
        入力:\n\
            ・恋人の聖地として認定されているロマンチックな場所\n\
            ・海水が澄んでいて綺麗\n\
            ・約束の丘展望台からの景色が良かった\n\
            ・エンジェルロードが感動的だった\n\
            ・小豆島の定番スポット\n\
            ・穏やかでキレイな海に癒される\n\
            ・干潮時の時間帯に見られるきれいなエンジェルロード\n\
            ・無料の駐車場が近く便利\n\
        出力:\n\
            ・恋人の聖地として認定されているロマンチックな定番スポット\n\
            ・穏やかでキレイな海に癒される\n\
            ・干潮時の時間帯に見られるきれいで感動的なエンジェルロード\n\
            ・無料の駐車場が近く便利\n\
        \n\
    それでは次の入力に対して同様に出力を行ってください\n'
            
    for topic in topics:
        topic = topic.split('\n')
        for i,t in enumerate(topic):
            if i==0:
                prompt += '            '+t
            else:
                prompt += '            ・'+t
            prompt += '\n'
    messages.append({"role": "user", "content":prompt})
    prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)

    return prompt

def find_connected_groups(matrix):
    def dfs(node, group):
        visited[node] = True
        group.append(node)
        for neighbour, isConnected in enumerate(matrix[node]):
            if isConnected and not visited[neighbour]:
                dfs(neighbour, group)

    n = len(matrix)
    visited = [False] * n
    groups = []

    for node in range(n):
        if not visited[node]:
            group = []
            dfs(node, group)
            groups.append(group)

    return groups

def extract_feature(pos_topics, device):
    topic_all = []
    for topic in pos_topics:
        #print(topic)
        topic = topic.split('\n')
        for t in topic:
            t = t.replace('<|im_end|>', '')
            if len(t)!=0:topic_all.append(t)
    #print(topic_all)
    feature_extractor = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual', device=f'cuda:{device}')
    features = feature_extractor.encode(topic_all)
    
    return topic_all, features

def groupby_sim(features, thresh=0.5):
    cos_sim = cosine_similarity(features)
    graph = cos_sim>=thresh
    group = find_connected_groups(graph)
    return group

def summarize_topic(group_topics, llm):
    messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
    
    prompt = '短文から共通のトピックを, 一文程度でまとめてください.次は例です.\n\
            例1:\n\
            入力:\n\
                坂道がきつい。\n\
                坂道が急で暑い\n\
            出力:\n\
                坂道がキツい\n\
            例2:\n\
            入力:\n\
                四季折々の花を楽しむことができる庭園があります。 \n\
                日本庭園は広大で、自然と人の手が調和した第一級の公園です。\n\
            出力:\n\
                日本庭園は広大で, 四季折々の花を楽しむことができる一級の庭園である\n\
            例3:\n\
            入力:\n\
                絶景の眺めがたくさんある。\n\
                観光を満喫できた。\n\
                ロープウェイの絶景が素晴らしい\n\
            出力:\n\
                ロープウェイなど絶景の眺めがたくさんある\n\
            \n\
            それでは次の入力に対して同様に出力を行ってください\n'
            
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    for topic in group_topics:
        prompt+=topic
        prompt+='\n'
    messages.append({"role": "user", "content":prompt})
    prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    prompts = [prompt]
    outputs = llm.generate(prompts, sampling_params)
    for k,output in enumerate(outputs):
        generated_text = output.outputs[0].text
    return generated_text
    
def make_final_topic(topic_all, topic_groups, llm):
    final_topics = []
    for group in topic_groups:
        
        if len(group)==1:
            final_topics.append(topic_all[group[0]])

        else:
            group_topics = []
            for g in group:
                group_topics.append(topic_all[g])
            #print(group_topics)
            summary_topic = summarize_topic(group_topics, llm)
            #summary_topic = ''
            final_topics.append(summary_topic)
            
    return final_topics
        
        
        
        
        
    

if __name__=='__main__':
    topic_summary = pd.read_csv('../data/topic_summary/topic_summary.csv', names=['index', 'spot', 'posneg', 'cluster_label', 'topic'])
    kagawa_popular_spots = ['金刀比羅宮', '栗林公園', 'エンジェルロード', 'レオマリゾート', '丸亀城', '瀬戸大橋（香川県坂出市）', '寒霞渓ロープウェイ', '道の駅\u3000小豆島オリーブ公園', '屋島', '二十四の瞳映画村', '銭形砂絵「寛永通宝」', '史跡高松城跡（玉藻公園）', '国営讃岐まんのう公園', '新屋島水族館', '瀬戸大橋記念公園', 'さぬきこどもの国', '地中美術館', 'マルキン醤油記念館', '直島諸島', 'サンポート高松']
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", dtype='float16', trust_remote_code=True)
    prompts = []
    index = []
    post_processed_topics_sentence = defaultdict(partial(defaultdict, list))
    post_processed_topics_gpt = defaultdict(partial(defaultdict, list))
    for spot in kagawa_popular_spots:
        pos_topics = topic_summary[(topic_summary['spot']==spot) & (topic_summary['posneg']=='pos')]['topic'].values
        neg_topics = topic_summary[(topic_summary['spot']==spot) & (topic_summary['posneg']=='neg')]['topic'].values
        
        pos_topic_all, pos_features = extract_feature(pos_topics, device=0)
        #print('pos', pos_topic_all, cosine_similarity(pos_features))
        pos_groups = groupby_sim(pos_features, thresh=0.7)
        #print('pos_groups', pos_groups)
        pos_final_groups = make_final_topic(pos_topic_all,pos_groups, llm)
        post_processed_topics_sentence[spot]['pos'] = pos_final_groups
        
        neg_topic_all, neg_features = extract_feature(neg_topics, device=0)
        #print('neg', neg_topic_all, cosine_similarity(neg_features))
        neg_groups = groupby_sim(neg_features, thresh=0.7)
        #print('neg_groups', neg_groups)
        neg_final_groups = make_final_topic(neg_topic_all,neg_groups, llm)
        post_processed_topics_sentence[spot]['neg'] = neg_final_groups
        
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]

        pos_prompt = make_prompt(pos_topics, llm)
        prompts.append(pos_prompt)
        index.append((spot, 'pos'))
        neg_prompt = make_prompt(neg_topics, llm)
        prompts.append(neg_prompt)
        index.append((spot, 'neg'))
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        
    outputs = llm.generate(prompts, sampling_params)
    for k,output in enumerate(outputs):
        spot, posneg = index[k]
        prompt = output.prompt
        generated_text = output.outputs[0].text
        #print('prompt', prompt)
        #print('generated text', generated_text)
        post_processed_topics_gpt[spot][posneg] = generated_text
        
    with open('../data/pp_topic/post_processed_topics_sentence.pkl', 'wb') as f:
        pickle.dump(post_processed_topics_sentence, f)
        
    with open('../data/pp_topic/post_processed_topics_sentence_gpt.pkl', 'wb') as f:
        pickle.dump(post_processed_topics_gpt, f)