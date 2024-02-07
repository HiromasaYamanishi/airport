import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

df = pd.read_csv('/home/yamanishi/project/airport/src/data/review_food_all_period_.csv')
reviews = df['review'].values
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

if torch.cuda.is_available():
    model = model.to("cuda")

for review in reviews[:100]:
    print('original review', review)
    #text =f"次のレビューのキーワードを3つ、この人が感じたいい点、悪い点をキーワードで教えてください レビュー: {review}"
    text = f"次のレビューは男性が書いたものか女性が書いたものか予測して下さい。また、予測の理由を教えて下さい"
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )


    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
    print(output)