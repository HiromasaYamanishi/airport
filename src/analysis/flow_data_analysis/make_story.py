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
story="""
0時(高松空港到着-8.0時間後)から7時(高松空港到着-1.0時間後)までは高松市にいてそこにある観光地は高松市レンタサイクル 丸亀町グリーン 高松市美術館, 飲食店は一鶴・高松店 さぬきうどん職人めりけんや 高松駅前店 天勝 本店です ホテルは14件あります
9時(高松空港到着0.0時間後)から8時(高松空港到着0.0時間後)までは高松市にいてそこにある観光地は, 飲食店はです ホテルは0件あります
10時(高松空港到着1.0時間後)から10時(高松空港到着1.0時間後)までは三豊市にいてそこにある観光地はサンポート高松, 飲食店はです ホテルは6件あります
11時(高松空港到着2.0時間後)から11時(高松空港到着2.0時間後)までは観音寺市にいてそこにある観光地は, 飲食店はです ホテルは0件あります
22時(高松空港到着13.0時間後)から22時(高松空港到着13.0時間後)までは観音寺市にいてそこにある観光地は讃岐路野天風呂　湯屋　琴弾廻廊 道の駅 ことひき, 飲食店はです ホテルは0件あります
"""
for review in reviews[:100]:
    #print('original review', review)
    text =f"次の人の旅行の行動を各時刻でどこで何をしているか推測して下さい: {story}"
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
    exit()