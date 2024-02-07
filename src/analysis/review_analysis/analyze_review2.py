import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import pickle
from tqdm import tqdm
import time


#df_kagawa = df[df['pref']=='香川県']
# tokenizer = AutoTokenizer.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
# messages.append({"role": "user", "content": f"このレビューはポジティブですかネガティブですか。またそう思う原因の部分をキーワードで抜き出してください。また、春夏秋冬の季節性のあるトピックがあれば抜き出してください {reviews[0]}"})

# prompt = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)

# pipe(prompt, max_new_tokens=100, do_sample=False, temperature=0.0, return_full_text=False)

# pipe(prompt, max_new_tokens=100, do_sample=False, temperature=0.0, return_full_text=False)

from vllm import LLM, SamplingParams
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ind', type=int, default=0)
args = parser.parse_args()

df = pd.read_csv('/home/yamanishi/project/airport/src/data/review_all_period_.csv')
#df = df[df['pref']=='香川県']
#df = df[:500]
reviews = df['review'].values
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
llm = LLM(model="lightblue/qarasu-14B-chat-plus-unleashed", trust_remote_code=True)
#llm = LLM(model="elyza/ELYZA-japanese-Llama-2-13b", trust_remote_code=True,)

ind = args.ind
div = 7
chunk=len(df)//div
save_output = {}
start = chunk*ind
if ind<div:
    end = chunk*(ind+1)
else:
    end = len(df)
    
df_target = df[start:end]




prompts = []
batch_size = 500
for i in tqdm(range(0, len(df_target), batch_size)):
    prompts = []
    for j in range(batch_size):
        if i+j>=len(df_target):continue
        messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
        messages.append({"role": "user", "content": f"このレビューがポジティブかネガティブか教えてください。また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで、ポジティブな点: 「〇〇な〇〇」「〇〇が〇〇」ネガティブな点: 「〇〇な〇〇」「〇〇が〇〇」のように抜き出してください。存在しなけれれば出力しなくて良いです\
                        レビュー: {df_target['review'].values[i+j]}。"})
        prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    outputs = llm.generate(prompts, sampling_params)
    for k,output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        save_output[start+i+k] = generated_text
        #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        #print(f"Generated text: {generated_text!r}")
        
       # save_output[chunk*ind+i] = generated_text
print(save_output)
with open(f'../data/review/goodbad_all_{ind}.pkl', 'wb') as f:
    pickle.dump(save_output, f)