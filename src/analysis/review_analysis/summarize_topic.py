import pickle
import numpy as np
from vllm import LLM, SamplingParams
import argparse
import pandas as pd
import os
from tqdm import tqdm

def summarize_topic(llm, clustered_sentence,summarize='general'):
    # summarize: [general or detail]
    cluster = clustered_sentence['clustering_labels']
    sentences = clustered_sentence['sentence']
    prompts = []
    meaningful_cluster = [i for i in np.unique(cluster) if i!=-1]
    for cluster_label in meaningful_cluster:
        #if cluster_label==-1:continue
        sentence_cluster = [sentences[i] for i in range(len(cluster)) if cluster[i]==cluster_label]
        
        if summarize=='general':
            chosen_sentence_cluster = np.random.choice(sentence_cluster, min(len(sentence_cluster), 20), replace=False)
        elif summarize=='detail':
            chosen_sentence_cluster = np.random.choice(sentence_cluster, min(len(sentence_cluster), 30), replace=False)
        
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        
        if summarize=='general':
            prompt = '短文から多数派のトピックのみを抜き出し, 一文程度でまとめてください.次は例です.\n\
例1:\n\
入力:\n\
    石段がきつくてたいへんです。\n\
    混雑しており前へ進まず。階段をひたすら上がらなければならない。\n\
    階段数が多い。\n\
    800段を超える石段は少し疲れます。\n\
    こんぴらさんの階段はきつくて足にきます。\n\
    道のりが長く、きつい。\n\
    観光化されている。\n\
    この角を曲がったらそこかな。階段を登らないと行けない。\n\
    子供連れにはあまりオススメできません\n\
出力:\n\
    800段を超える階段がキツくて大変\n\
例2:\n\
入力:\n\
    混雑する時間帯 \n\
    混雑する可能性がある\n\
    混雑\n\
    渋滞が発生する\n\
    賑やかだった\n\
    GWで混んでいた\n\
    人がいっぱい\n\
    人が多すぎた\n\
    人が多かった\n\
出力:\n\
    GWや時間によっては混雑する\n\
例3:\n\
入力:\n\
    猿の子連れがいてチヨットびっくり\n\
    山頂で大勢のサルが寝転がっている\n\
    ウリ坊が出現したり、展望台で猿の大群が移動したりして嬉しいサプライズがあった\n\
    表は猿が多く、道には猿の糞が多く踏まないよう注意する必要がある\n\
    野生の猿がいて間近で見ることが出来ました\n\
    山の上にあり、野生の猿がいる\n\
    途中の道路\n\
出力:\n\
    ウリや猿など野生の動物を見ることができる\n\
\n\
それでは次の入力に対して同様に出力を行ってください\n'
        elif summarize=='detail':
            prompt='短文から多数派のトピックのみを選択し, 50文字から100文字でまとめてください. 次のように2段階で行ってください.\n\
例:\n\
第一段階では, まず多数派のトピックとそれに対応する文章を選びます.\n\
入力:\n\
\n\
    癒しの場所。\n\
    のどかな雰囲気. 広い空間でのんびり過ごせる\n\
    解放された建物\n\
    自然に溶け込む非日常の空間。\n\
    リフレッシュされる。\n\
    周囲から隔離されて落ち着いた雰囲気。\n\
    中国人の留学生と来た。\n\
    マイナスイオンで心や体も癒されます\n\
    ベンチがあり, 休憩できる\n\
    都会の喧騒を忘れることができる\n\
    日常の嫌なことを忘れています\n\
\n\
ここで多数派のトピックは「リラックス・リフレッシュできる」ということです。\n\
第一段階の出力は, 多数派のトピックと対応するトピックで\n\
\n\
    癒しの場所。\n\
    のどかな雰囲気. 広い空間でのんびり過ごせる\n\
    自然に溶け込む非日常の空間。\n\
    リフレッシュされる。\n\
    周囲から隔離されて落ち着いた雰囲気。\n\
    マイナスイオンで心や体も癒されます\n\
    ベンチがあり, 休憩できる\n\
    都会の喧騒を忘れることができる\n\
    日常の嫌なことを忘れています\n\
\n\
第二段階では第一段階の出力を複数の文で要約します.例の第一段階の短文の集合を要約すると\n\
\n\
「周囲から隔離されたリラックスできる雰囲気の場所です. 自然に溶け込む非日常の空間の中で都会の喧騒や日常を忘れ、リフレッシュすることができます.」\n\
\n\
となります.よって出力は\n\
\n\
出力:\n\
「周囲から隔離されたリラックスできる雰囲気の場所です. 自然に溶け込む非日常の空間の中で都会の喧騒や日常を忘れ、リフレッシュすることができます.」です\n\
\n\
それでは次の入力の短文の集合に対して同様に第一段階, 第二段階の処理を行い, 第二段階の次の最終的な要約された文章のみを出力してください.第一段階の処理結果は出力しないでください.\n\
入力:\n\
\n'
        elif summarize=='simple':
            chosen_sentence_cluster = np.random.choice(sentence_cluster, min(len(sentence_cluster), 20), replace=False)
            prompt="次に与える短文の集合から多数派のトピックに該当する短文の内容をつなぎ合わせて100文字程度で要約してください.\n"
                              
        for sent in chosen_sentence_cluster:
            prompt += '    '+sent
            prompt += '\n'
        prompt+='出力:\n'
        messages.append({"role": "user", "content":prompt})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    
        
    outputs = llm.generate(prompts, sampling_params)
    topics = []
    for k,output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        topics.append(generated_text)
        
    return topics, meaningful_cluster
    
        
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--extra_suffix', type=str, default='')
    parser.add_argument('--summarize', type=str, default='general')
    parser.add_argument('--df_path', type=str, default='')
    args = parser.parse_args()


    # kagawa_popular_spots = ['金刀比羅宮', '栗林公園', 'エンジェルロード', 'レオマリゾート', '丸亀城', '瀬戸大橋（香川県坂出市）', '寒霞渓ロープウェイ', '道の駅\u3000小豆島オリーブ公園', '屋島', '二十四の瞳映画村', '銭形砂絵「寛永通宝」', '史跡高松城跡（玉藻公園）', '国営讃岐まんのう公園', '新屋島水族館', '瀬戸大橋記念公園', 'さぬきこどもの国', '地中美術館', 'マルキン醤油記念館', '直島諸島', 'サンポート高松']
    # df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/df_jalan_kagawa.csv')
    # kagawa_popular_spots = df_kagawa[df_kagawa['review_count']>20]['spot_name'].values
    
    df = pd.read_csv(args.df_path)
    if 'spot' in df.columns:
        spots = np.unique(df['spot'].values)
    elif 'hotel_name' in df.columns:
        spots = np.unique(df['hotel_name'].values)
    elif 'food' in df.columns:
        spots = np.unique(df['food'].values)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", dtype='float16', trust_remote_code=True)
    
    for spot in tqdm(spots):   
        save_name = args.suffix+'_'+args.extra_suffix
        if os.path.exists(f'../data/clustering/review/{spot}/{save_name}_neg_cluster_sentence.pkl'):
            with open(f'../data/clustering/review/{spot}/{save_name}_neg_cluster_sentence.pkl', 'rb') as f:
                neg_clustered_sentence = pickle.load(f)

            topics_neg, cluster_label = summarize_topic(llm, neg_clustered_sentence, summarize=args.summarize)
            cluster_num = len(cluster_label)
            df = pd.DataFrame({'spot': [spot for _ in range(cluster_num)],
                            'posneg': ['neg' for _ in range(cluster_num)],
                            'cluster': list(cluster_label),
                            'topics': topics_neg})
            df.to_csv(f'../data/topic_summary/topic_summary_{args.suffix}_{args.extra_suffix}_{args.summarize}.csv', mode='a', header=False)
        
        if os.path.exists(f'../data/clustering/review/{spot}/{save_name}_pos_cluster_sentence.pkl'):
            with open(f'../data/clustering/review/{spot}/{save_name}_pos_cluster_sentence.pkl', 'rb') as f:
                pos_clustered_sentence = pickle.load(f)

            topics_pos, cluster_label = summarize_topic(llm, pos_clustered_sentence,summarize=args.summarize)
            cluster_num = len(cluster_label)
            df = pd.DataFrame({'spot': [spot for _ in range(cluster_num)],
                            'posneg': ['pos' for _ in range(cluster_num)],
                            'cluster': list(cluster_label),
                            'topics': topics_pos})
            
            df.to_csv(f'../data/topic_summary/topic_summary_{args.suffix}_{args.extra_suffix}_{args.summarize}.csv', mode='a', header=False)
    exit()
    
    llm = LLM(model="elyza/ELYZA-japanese-Llama-2-13b", dtype='bfloat16', trust_remote_code=True)
    
    for spot in kagawa_popular_spots:   
        save_name = spot+'_'+args.suffix+args.extra_suffix
        
        with open(f'../data/clustering/review/{save_name}_neg_cluster_sentence.pkl', 'rb') as f:
            neg_clustered_sentence = pickle.load(f)

        topics_neg, cluster_label = summarize_topic(llm, neg_clustered_sentence, summarize=args.summarize)
        cluster_num = len(cluster_label)
        df = pd.DataFrame({'spot': [spot for _ in range(cluster_num)],
                           'posneg': ['neg' for _ in range(cluster_num)],
                           'cluster': list(cluster_label),
                           'topics': topics_neg})
        df.to_csv(f'../data/topic_summary/topic_summary_{args.suffix}_{args.extra_suffix}_{args.summarize}.csv', mode='a', header=False)
        
        with open(f'../data/clustering/review/{save_name}_pos_cluster_sentence.pkl', 'rb') as f:
            pos_clustered_sentence = pickle.load(f)

        topics_pos, cluster_label = summarize_topic(llm, pos_clustered_sentence,summarize=args.summarize)
        cluster_num = len(cluster_label)
        df = pd.DataFrame({'spot': [spot for _ in range(cluster_num)],
                           'posneg': ['pos' for _ in range(cluster_num)],
                           'cluster': list(cluster_label),
                           'topics': topics_pos})
        
        df.to_csv(f'../data/topic_summary/topic_summary_{args.suffix}_{args.extra_suffix}_{args.summarize}.csv', mode='a', header=False)
        
        
        