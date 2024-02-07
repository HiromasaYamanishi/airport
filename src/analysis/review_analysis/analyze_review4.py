import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import pickle
from tqdm import tqdm
import time
from vllm import LLM, SamplingParams
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ind', type=int, default=0)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--div', type=int, default=0)
    parser.add_argument('--data_type', type=str, default='spot')
    parser.add_argument('--df_path', type=str, )
    args = parser.parse_args()

    #df = pd.read_csv('/home/yamanishi/project/airport/src/data/review_all_period_.csv')
    df = pd.read_csv(args.df_path).reset_index()

    reviews = df['review'].values
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", dtype='float16', trust_remote_code=True)
    #llm = LLM(model="elyza/ELYZA-japanese-Llama-2-13b", trust_remote_code=True,)

    ind = args.ind
    div = args.div
    #div = 3
    chunk=len(df)//div
    save_output = {}
    start = chunk*ind
    if ind<div:
        end = chunk*(ind+1)
    else:
        end = len(df)
        
    df_target = df[start:end]


    prompt_spot = "このレビューがポジティブかネガティブか教えてください。\n\
        また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで出力してください。\n\
        例1:\n\
        レビュー: 深夜に女性専用ラウンジを利用しました。リクライニングソファがあり寝る事が出来ます。静かで雰囲気は良かったけど暖房が効きすぎているのか暑くて辛かったです。\n\
        出力:\n\
            ポジティブな点: 「リクライニングソファで寝ることができる」「静かで雰囲気が良い」\n\
            ネガティブな点: 「暖房が効きすぎていて暑い」\n\
        例2:\n\
        レビュー: 函館山からの夜景を楽しみに行ったのに2回とも雲がかかり上からの景色は見ることが出来なかった。ロープウェイで下降する時少し見えただけでした。\n\
        出力:\n\
            ポジティブな点: なし\n\
            ネガティブな点: 「雲で夜景が見えなかった」\n\
        例3:\n\
        レビュー: 平日の昼過ぎに訪れました。駅からの道のりは少しわかりにくいですが迷うことなく到着。すぐに鳳凰堂内部見学のため観覧券販売所を見つけ行列の後ろに。１回に５０人程度が入れるのかな。並んだ時に次の時間を買うことができました。鳳凰堂の中に入ってビックリしたのはライトアップされた阿弥陀如来坐像の煌びやかさとたくさんの菩薩像です。２０分程度の説明があっという間でした。仏像を前に身が引き締まる思いです\n\
        出力:\n\
            ポジティブな点: 「ライトアップされた阿弥陀如来坐像の煌びやかさ」「たくさんの菩薩像」「説明があっというまに感じるほど面白い」\n\
            ネガティブな点: 「駅からの道が分かりにくい」\n\
    それでは次のレビューに対して同様に出力してください。\n\
    レビュー: {}。"

    prompt_food = "このレビューがポジティブかネガティブか教えてください。\n\
        また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで出力してください。\n\
        例1:\n\
        レビュー: おいしいせんべい,瓦せんべいで人気の和菓子店です。いろいろな大きさがあって、たくさんお土産に買って帰りました。癖になる美味しさです。\n\
        出力:\n\
            ポジティブな点: 「瓦せんべいで人気」「いろいろなお土産がある」「癖になる美味しさのせんべい」\n\
            ネガティブな点: なし\n\
        例2:\n\
        レビュー: お店の方のかけ声と音楽でとっても楽しかったです！すごく混んでました、予約必須です。うどん作りしてうどん食べてお土産もらって大満足です\n\
        出力:\n\
            ポジティブな点: 「お店の方のかけ声と音楽が楽しかった」「うどん食べてお土産もらって満足して」\n\
            ネガティブな点: 「すごく混んでいた」「予約必須」\n\
        例3:\n\
        レビュー: こしがしっかりあるのが好きな方にはもの足りないかな？サイドメニューは豊富でした。味と値段も納得でした\n\
        出力:\n\
            ポジティブな点: 「サイドメニューが豊富」「納得の値段と味」\n\
            ネガティブな点: 「こしが足りない」\n\
    それでは次のレビューに対して同様に出力してください。\n\
    レビュー: {}。"

    prompt_hotel = "このレビューがポジティブかネガティブか教えてください。\n\
        また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで出力してください。\n\
        例1:\n\
        レビュー:思ったより歓楽街より遠かったです。第一目的の場所は近かったのですが…。部屋もお風呂も朝食も価格対で考えると満足のいくものでした。が、コンビニくらいは近くに欲しいかなぁ。なんとなくロビーの雰囲気が寂しかったです\n\
        出力:\n\
            ポジティブな点: 「部屋もお風呂も朝食も価格対で満足」\n\
            ネガティブな点: 「思ったより歓楽街より遠い」「コンビニが近くにない」「ロビーの雰囲気が寂しい」\n\
        例2:\n\
        レビュー:一泊二食付きプランを利用させて頂きましたが、夕食を食べに行くのに車で１０分ほど掛かるのと、朝食の量が少しもの足りませんでした。良かったところは、ホテルのスポーツジムを利用出来たところです。また、利用したく思ってます。\n\
        出力:\n\
            ポジティブな点: 「ホテルのスポーツジムを利用出来た」\n\
            ネガティブな点: 「夕食を食べに行くのに車で10分ほど掛かる」「朝食の量が少し物足りない」\n\
        例3:\n\
        レビュー:フロントの方の対応がとてもにこやかで室内も気持ちよく、泊まってよかったねと話していました。朝のバイキングも内容が豊富でゆっくり食事ができて満足です。\n\
        出力:\n\
            ポジティブな点: 「フロントの方の対応がにこやか」「バイキングの内容が豊富」「朝食をゆっくり食事ができる」\n\
            ネガティブな点: なし\n\
    それでは次のレビューに対して同様に出力してください。\n\
    レビュー: {}。"

    prompts = []
    batch_size = 100
    for i in tqdm(range(0, len(df_target), batch_size)):
        prompts = []
        for j in range(batch_size):
            if i+j>=len(df_target):continue
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            if args.data_type == 'spot':
                messages.append({"role": "user", "content": prompt_spot.format(df_target['review'].values[i+j])})
            elif args.data_type == 'food':
                messages.append({"role": "user", "content": prompt_food.format(df_target['review'].values[i+j])})
            elif args.data_type == 'hotel':
                messages.append({"role": "user", "content": prompt_hotel.format(df_target['review'].values[i+j])})
            prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params)
        for k,output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            save_output[start+i+k] = generated_text
            #print(df_target['review'].values[i+k])
            #print('generated text', generated_text)
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            #print(f"Generated text: {generated_text!r}")
            
        # save_output[chunk*ind+i] = generated_text
    print(len(save_output))
    with open(f'../data/review/goodbad_all_{args.suffix}_{ind}.pkl', 'wb') as f:
        pickle.dump(save_output, f)