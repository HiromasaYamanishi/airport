import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import pickle
from tqdm import tqdm


#df_kagawa = df[df['pref']=='香川県']
# tokenizer = AutoTokenizer.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
# messages.append({"role": "user", "content": f"このレビューはポジティブですかネガティブですか。またそう思う原因の部分をキーワードで抜き出してください。また、春夏秋冬の季節性のあるトピックがあれば抜き出してください {reviews[0]}"})

# prompt = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)

# pipe(prompt, max_new_tokens=100, do_sample=False, temperature=0.0, return_full_text=False)

# tokenizer = AutoTokenizer.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
# messages.append({"role": "user", "content": "イギリスの首相は誰ですか？"})

# prompt = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)

# pipe(prompt, max_new_tokens=100, do_sample=False, temperature=0.0, return_full_text=False)

from vllm import LLM, SamplingParams
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ind', type=int, default=0)
args = parser.parse_args()

df = pd.read_csv('/home/yamanishi/project/airport/src/data/review_all_period_.csv')
df = df[df['pref']=='香川県']
reviews = df['review'].values
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", trust_remote_code=True)

ind = args.ind
div = 4
chunk=len(df)//div
save_output = {}
if ind<div:
    for i,review in tqdm(enumerate(df['review'].values[chunk*ind:chunk*(ind+1)])):
        #review = 'はじめての四国旅行の最終日に立ち寄りました。高松市の中心地にほど近い場所に、大変広く整備された公園があり感心しました。日本庭園を主力として、各所に見どころが満載でした。紅葉の最盛期でしたが、園内には紅葉している木が多くは無く、見頃の木の近くでは多くの方が写真を撮っていました。池が多く、水面に映る風景も趣がありました。祭日の午前中でしたが、観光客が溢れることも無く、マイペースでの見学ができました'
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        messages.append({"role": "user", "content": f"次のレビューから感情を表す単語を全て抜き出して。その後(感情, 感情の対象)のペアを作成して\
                         レビュー: {review}。"})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts = [prompt]

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            print(f"Generated text: {generated_text!r}")
            
            save_output[chunk*ind+i] = generated_text

else:
    for i,review in tqdm(enumerate(df['review'].values[chunk*ind:])):
        #review = 'はじめての四国旅行の最終日に立ち寄りました。高松市の中心地にほど近い場所に、大変広く整備された公園があり感心しました。日本庭園を主力として、各所に見どころが満載でした。紅葉の最盛期でしたが、園内には紅葉している木が多くは無く、見頃の木の近くでは多くの方が写真を撮っていました。池が多く、水面に映る風景も趣がありました。祭日の午前中でしたが、観光客が溢れることも無く、マイペースでの見学ができました'
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        messages.append({"role": "user", "content": f"このレビューから感情を1単語で全て抜き出してください\
                         レビュー: {review}。"})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts = [prompt]

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            
            save_output[chunk*ind+i] = generated_text
        
with open(f'../data/review/goodbad_rev_{ind}.pkl', 'wb') as f:
    pickle.dump(save_output, f)