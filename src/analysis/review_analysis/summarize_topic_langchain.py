import pickle
import numpy as np
from vllm import LLM, SamplingParams
import argparse
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer
from  langchain import LLMChain, HuggingFacePipeline, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

def summarize_map_reduce(llm, clustered_sentence):
    cluster = clustered_sentence['clustering_labels']
    sentences = clustered_sentence['sentence']
    meaningful_cluster = [i for i in np.unique(cluster) if i!=-1]
    topics = []
    for cluster_label in meaningful_cluster:
        #if cluster_label==-1:continue
        sentence_cluster = [sentences[i] for i in range(len(cluster)) if cluster[i]==cluster_label]
        summary = map_reduce(llm, sentence_cluster)
        print(summary)
        topics.append(summary)
    return topics, meaningful_cluster

def map_reduce(llm, sentences):
    ## map
    template="""次に与える短文から多数派のトピックを認識し, 一文程度の要約のみを出力してください. 多数派でないトピックの内容は含めないでください.\n\
入力:'
    {docs}
出力:
    """
    
#     """短文から多数派のトピックのみを抜き出し, 一文程度でまとめてください.多数派でないトピックは含めないでください. 次は例です.\n\
# 入力例:
#     石段がきつくてたいへんです。\n\
#     混雑しており前へ進まず。階段をひたすら上がらなければならない。\n\
#     階段数が多い。\n\
#     800段を超える石段は少し疲れます。\n\
#     こんぴらさんの階段はきつくて足にきます。\n\
#     道のりが長く、きつい。\n\
#     観光化されている。\n\
#     この角を曲がったらそこかな。階段を登らないと行けない。\n\
#     子供連れにはあまりオススメできません\n\
# 出力例:
#     800段を超える石段の階段がキツくて大変。子供連れにはオススメできない\n\
# 例2:\n\
# 入力例:
#     駅からの道が分かりにくい
#     涼しい場所が少ないので、真夏は少しキツい
#     夏は暑くて日陰がなく、熱中症にならないように注意が必要です。
#     GWや暑い時期にセミがいる。
#     夏の暑さは注意が必要です。熱中症に注意しましょう。
#     蚊や虫よけ対策が必要な場所
# 出力例:
#     日陰がなく真夏は熱中症にならないように注意が必要. 蚊や虫除け対策が必要\n\
# 例3:\n\
# 入力例:
#     日本庭園や歴史博物館のようなものがお好きな方にお勧め
#     お茶を楽しむことができる
#     日本庭園は広大で回遊式で、四季折々の自然を堪能できる。
#     広大な敷地に素晴らしい日本庭園が広がっています。
#     日本庭園の手入れが行き届いており、四季折々の美しさが楽しめます。
#     秋の紅葉が美しい
#     日本古来の伝統を感じる。和の素晴らしさを実感できる。
# 出力例:
#     広大な敷地での日本庭園は手入れが行き届いている. 和の素晴らしさや四季折々の美しさが感じられる\n\
# \n\
# それでは次の入力に対して同様に出力を行ってください\n
# 入力:'
#     {docs}
# 出力:
#     """
    chunk_num = max(int(np.sqrt(len(sentences))), 10)
    summaries = []
    docs = ['' for _ in range(len(sentences)//chunk_num+1)]
    for i in range(len(sentences)):
        if '入力:' in sentences[i]:continue
        docs[i//chunk_num] +='    '
        docs[i//chunk_num] += sentences[i].replace('<|im_end|>\n', '')
        docs[i//chunk_num] += '\n'
        
    prompts = []
    for doc in docs:
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        messages.append({"role": "user", "content":template.format(docs=doc)})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
        
    outputs = llm.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    print('map outputs', outputs)
    ## reduce
    
    docs = ''
    for i,output in enumerate(outputs):
        if '入力:' in output:continue
        if i!=0:docs+='    '
        docs+=output.replace('<|im_end|>\n', '')
        docs+='\n'
        if len(docs)>=2000:break
    template="""次に与える短文から多数派のトピックを認識し, 多数派のトピックと一文程度の要約を出力してください. 多数派でないトピックの内容は含めないでください.\n\
入力:'
    {docs}
出力:
    """
    
#     """次の要約から多数派のトピックのみを抜き出し,詳細も適度に残して二文程度でまとめてください.多数派でないトピックは含めないでください. 次は例です.\n\
# 例1:\n\
# 入力例:\n\
#     石段がきつくてたいへんです。\n\
#     混雑しており前へ進まず。階段をひたすら上がらなければならない。\n\
#     階段数が多い。\n\
#     800段を超える石段は少し疲れます。\n\
#     こんぴらさんの階段はきつくて足にきます。\n\
#     道のりが長く、きつい。\n\
#     観光化されている。\n\
#     この角を曲がったらそこかな。階段を登らないと行けない。\n\
#     子供連れにはあまりオススメできません\n\
# 出力例:\n\
#     800段を超える石段の階段がキツくて大変。子供連れにはオススメできない\n\
# 例2:\n\
# 入力例:\n\
#     駅からの道が分かりにくい
#     涼しい場所が少ないので、真夏は少しキツい
#     夏は暑くて日陰がなく、熱中症にならないように注意が必要です。
#     GWや暑い時期にセミがいる。
#     夏の暑さは注意が必要です。熱中症に注意しましょう。
#     蚊や虫よけ対策が必要な場所
# 出力例:\n\
#     日陰がなく真夏は熱中症にならないように注意が必要. 蚊や虫除け対策が必要\n\
# 例3:\n\
# 入力例:\n\
#     日本庭園や歴史博物館のようなものがお好きな方にお勧め
#     お茶を楽しむことができる
#     日本庭園は広大で回遊式で、四季折々の自然を堪能できる。
#     広大な敷地に素晴らしい日本庭園が広がっています。
#     日本庭園の手入れが行き届いており、四季折々の美しさが楽しめます。
#     秋の紅葉が美しい
#     日本古来の伝統を感じる。和の素晴らしさを実感できる。
# 出力例:\n\
#     広大な敷地での日本庭園は手入れが行き届いている. 和の素晴らしさや四季折々の美しさが感じられる\n\
# \n\
# それでは次の入力に対して同様に出力を行ってください\n'
# 入力:
#     {docs}
# 出力:
#     """
    prompts = []
    messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
    #print(template.format(docs=docs))
    messages.append({"role": "user", "content":template.format(docs=docs)})
    prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    prompts.append(prompt)
        
    outputs = llm.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    print('reduce outputs', outputs)
    
    return outputs[0]
        
    ## reduce
        

def summarize_topic_chain(llm, clustered_sentence, args):
    cluster = clustered_sentence['clustering_labels']
    sentences = clustered_sentence['sentence']
    meaningful_cluster = [i for i in np.unique(cluster) if i!=-1]
    for cluster_label in meaningful_cluster:
        #if cluster_label==-1:continue
        sentence_cluster = [sentences[i] for i in range(len(cluster)) if cluster[i]==cluster_label]
        if args.chain_method == 'stuff':
            cat_sentences = ''
            for s in sentence_cluster:
                cat_sentences+=s
            
            docs = [Document(page_content=cat_sentences, metadata={})]  
            prompt_template = """次の中から多数派のトピックを選び, 適切に要約してください:
            "{text}"
            適切な要約:"""
            prompt = PromptTemplate.from_template(prompt_template)
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            summary = stuff_chain.run(docs)
            
        elif args.chain_method == 'map_reduce':
            chunk_num = 10
            docs = ['' for _ in range(len(sentence_cluster)//chunk_num+1)]
            for i in range(len(sentence_cluster)):
                if '入力:' in sentences[i]:continue
                docs[i//chunk_num] += sentences[i]
                
            docs = [Document(page_content=s, metadata={}) for s in docs]
                
            map_template ="""短文から多数派のトピックのみを抜き出し, 一文程度でまとめてください.次は例です.\n\
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
    800段を超える石段の階段がキツくて大変\n\
例2:\n\
入力:\n\
    混雑する時間帯 \n\
    混雑する可能性がある\n\
    渋滞が発生する\n\
    賑やかだった\n\
    GWで混んでいた\n\
    人がいっぱい\n\
    人が多すぎた\n\
    人が多かった\n\
出力:\n\
    GWや時間によっては混雑する. 渋滞が発生する\n\
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
    ウリや猿の大群など野生の動物を間近に見ることができる\n\
\n\
それでは次の入力に対して同様に出力を行ってください\n'
            {docs}
            """
                
            # map_template = """The following is a set of documents
            #             {docs}
            #             Based on this list of docs, please identify the main themes 
            #             Helpful Answer:"""
            map_prompt = PromptTemplate.from_template(map_template)
            map_chain = LLMChain(llm=llm, prompt=map_prompt)
            
            reduce_template = """これらの文章からから多数派のトピックのみを抜き出し, テーマを一文程度でまとめてください.次は例です.\n\
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
    800段を超える石段の階段がキツくて大変\n\
例2:\n\
入力:\n\
    混雑する時間帯 \n\
    混雑する可能性がある\n\
    渋滞が発生する\n\
    賑やかだった\n\
    GWで混んでいた\n\
    人がいっぱい\n\
    人が多かった\n\
出力:\n\
    GWや時間によっては混雑する. 渋滞が発生する\n\
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
    ウリや猿の大群など野生の動物を間近に見ることができる\n\
\n\
それでは次の入力に対して同様に出力を行ってください\n'
                {docs}"""
            # reduce_template = """The following is set of summaries:
            # {docs}
            # Take these and distill it into a final, consolidated summary of the main themes. 
            # Helpful Answer:"""
            reduce_prompt = PromptTemplate.from_template(reduce_template)
        
            reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

            # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain, document_variable_name="docs"
            )

            # Combines and iteratively reduces the mapped documents
            reduce_documents_chain = ReduceDocumentsChain(
                # This is final chain that is called.
                combine_documents_chain=combine_documents_chain,
                # If documents exceed context for `StuffDocumentsChain`
                collapse_documents_chain=combine_documents_chain,
                # The maximum number of tokens to group documents into.
                token_max=4000,
            ) 
            
            map_reduce_chain = MapReduceDocumentsChain(
                # Map chain
                llm_chain=map_chain,
                # Reduce chain
                reduce_documents_chain=reduce_documents_chain,
                # The variable name in the llm_chain to put the documents in
                document_variable_name="docs",
                # Return the results of the map steps in the output
                return_intermediate_steps=False,
            )
            summary = map_reduce_chain.run(docs)
            print(summary)
        
        elif args.chain_method == 'refine':
            docs = ['' for _ in range(len(sentence_cluster)//chunk_num+1)]
            for i in range(len(sentence_cluster)):
                docs[i//chunk_num] += sentences[i]
                
            docs = [Document(page_content=s, metadata={}) for s in docs]
                
            refine_template = (
                "Your job is to produce a final summary\n"
                "We have provided an existing summary up to a certain point: {existing_answer}\n"
                "We have the opportunity to refine the existing summary"
                "(only if needed) with some more context below.\n"
                "------------\n"
                "{text}\n"
                "------------\n"
                "Given the new context, refine the original summary in Italian"
                "If the context isn't useful, return the original summary."
            )
            refine_prompt = PromptTemplate.from_template(refine_template)
            chain = load_summarize_chain(
                llm=llm,
                chain_type="refine",
                question_prompt=prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",
            )
            summary = chain({"input_documents": docs}, return_only_outputs=True)
        
    return summary

    

def summarize_topic(llm, clustered_sentence,):
    cluster = clustered_sentence['clustering_labels']
    sentences = clustered_sentence['sentence']
    prompts = []
    meaningful_cluster = [i for i in np.unique(cluster) if i!=-1]
    for cluster_label in meaningful_cluster:
        #if cluster_label==-1:continue
        sentence_cluster = [sentences[i] for i in range(len(cluster)) if cluster[i]==cluster_label]
        
        chosen_sentence_cluster = np.random.choice(sentence_cluster, min(len(sentence_cluster), 20), replace=False)
        
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        
        prompt = '短文から多数派のトピックのみを抜き出し, テーマを一文程度でまとめてください.次は例です.\n\
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
                
        for sent in chosen_sentence_cluster:
            prompt +=sent
            prompt += '\n'
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
    parser.add_argument('--method', type=str, default='naive') # [naive, chain]
    parser.add_argument('--chain_method', type=str, default='stuff') # [stuff, map, refine] 
    args = parser.parse_args()


    kagawa_popular_spots = ['金刀比羅宮', '栗林公園', 'エンジェルロード', 'レオマリゾート', '丸亀城', '瀬戸大橋（香川県坂出市）', '寒霞渓ロープウェイ', '道の駅\u3000小豆島オリーブ公園', '屋島', '二十四の瞳映画村', '銭形砂絵「寛永通宝」', '史跡高松城跡（玉藻公園）', '国営讃岐まんのう公園', '新屋島水族館', '瀬戸大橋記念公園', 'さぬきこどもの国', '地中美術館', 'マルキン醤油記念館', '直島諸島', 'サンポート高松']
    #kagawa_popular_spots = ['金刀比羅宮']
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    
    if args.method == 'chain':
        model = 'lightblue/qarasu-14B-chat-plus-unleashed'
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                max_length=tokenizer.model_max_length,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
        llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
        
        
    elif args.method == 'naive':
        llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", dtype='float16', trust_remote_code=True)
        
    elif args.method == 'map_reduce':
        llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", dtype='float16', trust_remote_code=True)
      
    # split_docs = ['私は犬です', '私の父は犬です','私の母は犬です']
    # split_docs = [Document(page_content=s, metadata={}) for s in split_docs]  
    # chain = load_summarize_chain(llm, chain_type="refine")
    # print('chain', chain)
    # summary = chain.run(split_docs)
    # print('summry tmp', summary)
    
    for spot in kagawa_popular_spots:   
        save_name = spot+'_'+args.suffix+args.extra_suffix
        with open(f'../data/clustering/review/{save_name}_pos_cluster_sentence.pkl', 'rb') as f:
            pos_clustered_sentence = pickle.load(f)

        if args.method == 'naive':
            topics_pos, cluster_label = summarize_topic(llm, pos_clustered_sentence,)
        elif args.method == 'chain':
            topics_pos, cluster_label = summarize_topic_chain(llm, pos_clustered_sentence, args)
        elif args.method == 'map_reduce':
            topics_pos, cluster_label = summarize_map_reduce(llm, pos_clustered_sentence,)

        print(topics_pos)    
        cluster_num = len(cluster_label)
        df = pd.DataFrame({'spot': [spot for _ in range(cluster_num)],
                           'posneg': ['pos' for _ in range(cluster_num)],
                           'cluster': list(cluster_label),
                           'topics': topics_pos})
        
        df.to_csv(f'../data/topic_summary/topic_summary_{args.suffix}.csv', mode='a', header=False)
        
        with open(f'../data/clustering/review/{save_name}_neg_cluster_sentence.pkl', 'rb') as f:
            neg_clustered_sentence = pickle.load(f)

        if args.method == 'naive':
            topics_neg, cluster_label = summarize_topic(llm, neg_clustered_sentence,)
        elif args.method == 'chain':
            topics_neg, cluster_label = summarize_topic(llm, neg_clustered_sentence,)
        elif args.method == 'map_reduce':
            topics_neg, cluster_label = summarize_map_reduce(llm, neg_clustered_sentence,)
            
        cluster_num = len(cluster_label)
        df = pd.DataFrame({'spot': [spot for _ in range(cluster_num)],
                           'posneg': ['neg' for _ in range(cluster_num)],
                           'cluster': list(cluster_label),
                           'topics': topics_neg})
        df.to_csv(f'../data/topic_summary/topic_summary_{args.suffix}.csv', mode='a', header=False)
    
        
        
        