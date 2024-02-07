from sentence_transformers import SentenceTransformer
from transformers import MLukeTokenizer, LukeModel
from vllm import LLM as vllm_LLM
from vllm import SamplingParams
import torch
from typing import List

class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings).detach().numpy()
    
class LLM:
    def __init__(self, model_name="lightblue/qarasu-14B-chat-plus-unleashed"):
        self.llm = vllm_LLM(model=model_name, dtype='float16', trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
        
    def generate(self, prompts: List[str]):
        prompts_llm = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompts_llm.append(prompt)
        outputs = self.llm.generate(prompts_llm, self.sampling_params)
        generated_texts = []
        for k,output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
            
        return generated_texts