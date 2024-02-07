import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from copy import deepcopy
from vllm import LLM, SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForMaskedLM

class Decoder:
    def __init__(
        self,
        modelpath="lightblue/qarasu-14B-chat-plus-unleashed",
        lora_weight="",
        mask_embedding_sentence_template=None, #'This_passage_:_"*sent_0*"_means_in_one_word:"',
        avg=False,
        bf16 = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #self.tokenizer.padding_side = "left"  # Allow batched inference
        self.model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
            device_map="auto",
        )
        # self.model = PeftModel.from_pretrained(
        #     self.model,
        #     lora_weight,
        #     torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
        #     device_map='auto',
        # )
        #self.model = self.model.merge_and_unload()
        self.model.save_pretrained("./temp")
        #del self.model
        #vllm
        #self.llm = LLM(model="./temp",tokenizer=modelpath,dtype='float16' if bf16 == False else "bfloat16")
        self.llm = LLM(model="./temp", tokenizer=modelpath)
        print(self.llm.llm_engine.workers)
        #self.model = self.llm.llm_engine.workers[0].model #opt
        self.model = self.llm.llm_engine.driver_worker.model
        self.tokenizer = self.llm.llm_engine.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelpath, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"
        # self.model.eval()
        self.mask_embedding_sentence_template = mask_embedding_sentence_template
        self.avg = avg
        print(self.mask_embedding_sentence_template)

    def encode(self, raw_sentences, batch_size=32, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        #     batch = [[word.decode("utf-8") for word in s] for s in batch]

        # sentences = [" ".join(s) for s in batch]
        # input_sentences = [" ".join(s) for s in batch]
        # if max_length == 500:
        #     sentences = [
        #         self.tokenizer.decode(
        #             self.tokenizer.encode(s, add_special_tokens=False)[:max_length]
        #         )
        #         for s in sentences
        #     ]
        #     max_length = 512
        max_length = 512
        sentences = deepcopy(raw_sentences)

        if (
            # args.mask_embedding_sentence
            # and
            self.mask_embedding_sentence_template
            is not None
        ):
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = self.mask_embedding_sentence_template
            template = (
                template.replace("_", " ").replace("*sep+*", "").replace("*cls*", "")
            )

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in ".?\"'":
                    s += "."
                s = s.replace('"', "'")
                if len(s) > 0 and "?" == s[-1]:
                    s = s[:-1] + "."
                sentences[i] = template.replace("*sent 0*", s).strip()

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vocab_size = self.llm.llm_engine.workers[0].model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.llm.llm_engine.workers[0].scheduler_config.max_num_batched_tokens
        max_num_seqs = self.llm.llm_engine.workers[0].scheduler_config.max_num_seqs

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            batch = self.tokenizer.batch_encode_plus(
                sentences_batch,
                # return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=max_length is not None,
            )['input_ids']
            # Move to the correct device
            # for k in batch:
            #     batch[k] = batch[k].to(self.device) if batch[k] is not None else None
            #    # Get raw embeddings
            # batch = {k: v.to(self.device) for k,v in batch.items()}
            seqs = []
            for group_id in range(len(batch)):
                seq_data = SequenceData(list(batch[group_id]))
                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                )
                seqs.append(seq)

            input_tokens, input_positions, input_metadata = self.llm.llm_engine.workers[0]._prepare_inputs(seqs)
            num_layers = self.llm.llm_engine.workers[0].model_config.get_num_layers(self.llm.llm_engine.workers[0].parallel_config)
            with torch.no_grad():
                outputs = self.llm.llm_engine.workers[0].model.model( #opt
                # outputs = self.llm.llm_engine.workers[0].model.transformer( #falcon
                    input_ids=input_tokens,
                    positions=input_positions,
                    kv_caches=[(None, None)] * num_layers,
                    input_metadata=input_metadata,
                    cache_events=None,
                )
                outputs = outputs[:, -1, :]
            all_embeddings.extend(outputs.float().cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings
    
def get_embedding():
    tokenizer = AutoTokenizer.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("lightblue/qarasu-14B-chat-plus-unleashed", device_map="auto", trust_remote_code=True)
    df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/kagawa_review.csv')
    batch_size=50
    ind = np.random.randint(0, len(df_kagawa), size=batch_size)
    sentences = list(df_kagawa['review'][ind].values)
    print('sentence', )
    device='cuda'
    model = model.to(device)
    t_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    print('t_input')
    with torch.no_grad():
        last_hidden_state = model(**t_input, output_hidden_states=True).hidden_states[-1].to('cpu')
    
    weights_for_non_padding = t_input.attention_mask.to('cpu') * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
    sentence_embeddings = sentence_embeddings.float()
    print(sentence_embeddings)
    print(sentence_embeddings.shape)
    cos_sim = cosine_similarity(sentence_embeddings, sentence_embeddings)
    for i in range(10):
        j1 = np.random.randint(0, batch_size)
        j2 = np.random.randint(0, batch_size)
        print(f'文{j1}: {sentences[j1]}, 文{j2}: {sentences[j2]}, 類似度: {cos_sim[j1, j2]}')
        
def get_embedding_roberta():
    #tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base", use_fast=False)
    #tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    #model = AutoModelForMaskedLM.from_pretrained("rinna/japanese-roberta-base")
    df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/kagawa_review.csv')
    batch_size=50
    ind = np.random.randint(0, len(df_kagawa), size=batch_size)
    sentences = list(df_kagawa['review'][ind].values)
    feature_extractor = pipeline("feature-extraction",framework="pt",model="rinna/japanese-roberta-base", device=0)
    emb = feature_extractor(sentences,return_tensors = "pt")[0].numpy().mean(axis=0)
    print(emb.shape)
    cos_sim = cosine_similarity(emb, emb)
    for i in range(10):
        j1 = np.random.randint(0, batch_size)
        j2 = np.random.randint(0, batch_size)
        print(f'文{j1}: {sentences[j1]}, 文{j2}: {sentences[j2]}, 類似度: {cos_sim[j1, j2]}')
    
if __name__=='__main__':
    df_kagawa = pd.read_csv('/home/yamanishi/project/airport/src/data/kagawa_review.csv')
    sentences = df_kagawa['review'][:100]
    
    
    get_embedding_roberta()
    # model = Decoder()
    # embedding = model.encode(sentences)
    # print(embedding.shape)