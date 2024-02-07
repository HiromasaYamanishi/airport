import pickle
from vllm import LLM, SamplingParams
import pandas as pd
import copy
def directly_summarize_reviews(reviews, llm):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    original_prompt = 'レビューの集合を与えます.それらからポジティブな要素, ネガティブな要素を抽出し, 箇条書きで出力してください.\n\
例1:\n\
入力:\n\
    ８００段を越える石段で有名なこんぴらさんですが、実は、幼稚園にも満たないお子さんでも参拝できます。奥宮までは、１３００段以上のなりますが、３つの女の子がお父さんと手をつないで楽しい層に登ってきたのが印象的でした。\n\
    当日雨予報でしたが現地着いた頃には上がっていたので傘を杖代わりに嫁と共に幸福の黄色御守りゲットの為参拝。帰りは雨が本降りになってきて足元かなり滑るのでまあまあ大変でした。\n\
    参道の階段を700数段上がったところの本宮は迫力で、展望テラス？もすごく見通しがよく遠くまで見えました。そこからさらに600数段上がったところに奥宮があり、奥宮限定のお守りが買えるということで頑張って奥宮まで登りました。奥宮は小さなお宮でしたが風情がありました。\n\
    久しぶりの金比羅さんでした。夕方4時前に宿に着き、御朱印を貰いたくて急いで登り16時半少し前に着き、御朱印をいただく事が出来ましたが汗だくでした。本宮も奥社も16時半までですが奥社は朝も9時からなので、気をつけて予定を立てて下さい。\n\
    少し風が冷たく、肌寒い感じはしましたが、階段を登りだせば汗だくになりました。途中、焼き立てのお煎餅をつまみ食い、とっても美味しかったです！\n\
    前回6月にお参りしましたが、暑くて本宮までしか行けませんでした。登っても登っても、先が見えず、かなりしんどかったです。天気は良く、ところどころ紅葉も見れました。\n\
    沢山の階段をお土産屋さんを見ながら登りようやくお守りを購入出来ました。帰りの際お土産をゆっくり見たり足湯も満喫できました。\n\
    大変ではありましたが行った達成感があるので行って下さいそこでしか買えない御守りや御朱印も頂けます途中の神椿でのパフェも美味しいので食べて良かったです\n\
    雨上がりに初めてお参りに行きました。靴底のデザインがツルツル系でしたが滑ってしまい派手に転倒してしまいました。階段が急な所も有り、靴の選択と濡れている場合は十分な注意が必要です。\n\
出力:\n\
    ポジティブな点:\n\
        ・焼きたてのお煎餅や神椿でのパフェが美味しい\n\
        ・階段を登った本宮の迫力がある\n\
        ・展望テラスからの見晴らしが良い\n\
        ・限定のお守りが買える\n\
        ・足湯が満喫できる\n\
    ネガティブな点:\n\
        ・雨が降ると滑って危険\n\
        ・階段が長くてしんどい\n\
        ・奥社の空いている時間に気をつける必要がある\n\
それでは次の入力に対して同様に出力を行ってください\n\
入力:\n'
    # 15個づづレビューを処理する
    prompts = []
    chunk_size = 10
    for i in range((len(reviews)-1)//chunk_size+1):
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        prompt = copy.copy(original_prompt[:])
        for j in range(chunk_size*i, min(chunk_size*(i+1), len(reviews))):
            prompt+=reviews[j]
        messages.append({"role": "user", "content":prompt})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
        
    generated_topics = []
    outputs = llm.generate(prompts, sampling_params)
    for k,output in enumerate(outputs):
        generated_text = output.outputs[0].text
        generated_topics.append(generated_text)
    print('generated topics', generated_topics)
    #print(generated_topics)
    topic_summarize_original_prompt = 'これから施設の要約されたポジティブな文章の一覧とネガティブな文章の一覧を複数与えます.これらを要約して, 再び要約したポジティブの文章とネガティブな文章を出力してください.ただし，再び要約したポジティブな文章とネガティブな文章はそれぞれ多くても30個程度にしてください.\n\
例1:\n\
入力:\n\
    要約1:\n\
    ポジティブな点:\n\
        手入れが行き届いた素敵な庭園\n\
        時間があればお食事やお茶を楽しめます\n\
        ライトアップも見ることができました\n\
    ネガティブな点:\n\
        時間がないと楽しめない\n\
    要約2:\n\
    ポジティブな点:\n\
        夜桜ライトアップが美しかった\n\
    ネガティブな点:\n\
        駐車料金が高い\n\
        人が多い\n\
    要約3:\n\
    ポジティブな点:\n\
        公園内に印象に残る撮影スポットがたくさんある\n\
        日本三名園より優れていると言われている\n\
        お抹茶を飲みながら眺める庭が素晴らしい\n\
    ネガティブな点:\n\
        混雑している\n\
        駐車場が混雑している\n\
        庭園が広く、歩くのに時間がかかる\n\
出力:\n\
    ポジティブな点:\n\
        手入れが行き届いた素敵な庭園\n\
        お食事やお茶を楽しめます\n\
        撮影スポットが多い\n\
        夜桜のライトアップが綺麗\n\
    ネガティブな点:\n\
        駐車場の料金の高さと混雑\n\
        園内の混雑\n\
        園内が広く回るのに時間がかかる\n\
\n\
それでは次の要約群を要約して，ポジティブな文章とネガティブな文章を要約してください\n\
入力:\n'
    # hierachical summarization step1
    prompts = []
    summary_chunk_size = 10
    for i in range((len(generated_topics)-1)//summary_chunk_size+1):
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        topic_prompt = copy.copy(topic_summarize_original_prompt)
        for j in range(summary_chunk_size*i, min(summary_chunk_size*(i+1), len(generated_topics))):
            topic_prompt+=f'要約{j-summary_chunk_size*i+1}:\n'
            topic_prompt+=generated_topics[j]
        #print('step1 prompt', topic_prompt)
        messages.append({"role": "user", "content":topic_prompt})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
        
    summarized_topics = []
    outputs = llm.generate(prompts, sampling_params)
    for k,output in enumerate(outputs):
        generated_text = output.outputs[0].text
        summarized_topics.append(generated_text)
    print('summarized topic', summarized_topics)
        
    # hierachical summarization step2
    if len(summarized_topics)>1:
        prompts = []
        summary_chunk_size = 10
        for i in range((len(summarized_topics)-1)//summary_chunk_size+1):
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            topic_final_prompt = topic_summarize_original_prompt[:]
            for j in range(summary_chunk_size*i, min(summary_chunk_size*(i+1), len(summarized_topics))):
                topic_final_prompt+=f'要約{j-summary_chunk_size*i+1}:\n'
                topic_final_prompt+=summarized_topics[j]
            #print('step 2 prompt', topic_final_prompt)
            messages.append({"role": "user", "content":topic_final_prompt})
            prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)
        
    summarized_final_topics = []
    outputs = llm.generate(prompts, sampling_params)
    for k,output in enumerate(outputs):
        generated_text = output.outputs[0].text
        summarized_final_topics.append(generated_text)
    print('summarize final topic', summarized_final_topics)
    return summarized_final_topics

if __name__=='__main__':
    df = pd.read_csv('/home/yamanishi/project/airport/src/data/review_all_period_.csv')
    df_kagawa = df[df['pref']=='香川県']
    kagawa_popular_spots = ['金刀比羅宮', '栗林公園', 'エンジェルロード', 'レオマリゾート', '丸亀城', '瀬戸大橋（香川県坂出市）', '寒霞渓ロープウェイ', '道の駅\u3000小豆島オリーブ公園', '屋島', '二十四の瞳映画村', '銭形砂絵「寛永通宝」', '史跡高松城跡（玉藻公園）', '国営讃岐まんのう公園', '新屋島水族館', '瀬戸大橋記念公園', 'さぬきこどもの国', '地中美術館', 'マルキン醤油記念館', '直島諸島', 'サンポート高松']
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", dtype='bfloat16',gpu_memory_utilization=0.9, trust_remote_code=True)
    direct_summary = {}
    for spot in kagawa_popular_spots:
        df_kagawa_tmp_reviews = df_kagawa[df_kagawa['spot']==spot]['review'].values[:1000]
        
        summary = directly_summarize_reviews(df_kagawa_tmp_reviews, llm)
        direct_summary[spot] = summary
        
        df = pd.DataFrame({'spot': [spot],
                           'topics': [summary]})
        
        df.to_csv('../data/direct_summary/topic.csv', mode='a', header=False)
        
    with open('../data/direct_summary/direct_summary.pkl', 'wb') as f:
        pickle.dump(direct_summary, f)
        