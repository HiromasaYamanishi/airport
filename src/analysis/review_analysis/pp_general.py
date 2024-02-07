from models.language_models import LLM
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

class PostProcessor:
    def __init__(self):
        self.llm = LLM()
        
    def merge_group(self, df_simple, groups):
        df_merged = []
        for spot in groups.keys():
            df_simple_pos_tmp = df_simple[(df_simple['spot']==spot)&(df_simple['posneg']=='pos')].reset_index()
            pos_groups = groups[spot]['pos']
            pos_groups_rep = [p[0] for p in pos_groups]
            
            combined_topics = []
            for group in pos_groups:
                combined_topic = ' '.join(df_simple_pos_tmp.loc[df_simple_pos_tmp.index.isin(group), 'topics'])
                combined_topics.append(combined_topic)
            # print(pos_groups)
            # print(pos_groups_rep)
            # print(combined_topics)  
            df_simple_pos_rep = df_simple_pos_tmp.loc[pos_groups_rep]
            df_simple_pos_rep['combined_topic'] = combined_topics
            df_merged.append(df_simple_pos_rep)
                
            df_simple_neg_tmp = df_simple[(df_simple['spot']==spot)&(df_simple['posneg']=='neg')].reset_index()
            neg_groups = groups[spot]['neg']
            neg_groups_rep = [p[0] for p in neg_groups]
            
            combined_topics = []
            for group in neg_groups:
                combined_topic = ' '.join(df_simple_neg_tmp.loc[df_simple_neg_tmp.index.isin(group), 'topics'])
                combined_topics.append(combined_topic)
            df_simple_neg_rep = df_simple_neg_tmp.loc[neg_groups_rep]
            df_simple_neg_rep['combined_topic'] = combined_topics
            df_merged.append(df_simple_neg_rep)
            
        df_merged = pd.concat(df_merged)
        df_merged = df_merged.drop('level_0', axis=1)
        df_merged = df_merged.reset_index()
        return df_merged
    
    def negative_check(self, df_general: pd.DataFrame):
        '''
        df_generalにおけるnegの文章が本当にnegかをチェックする
        '''
        prompt = "次の文章がポジティブかネガティブか判定してください．次は例です\n\
例1:\n\
入力:\n\
    人気のスポットで混雑する場所にあります。\n\
出力:\n\
    ネガティブ\n\
例2:\n\
入力:\n\
    古民家風の展示物や外観が古民家であるなど、古民家を生かした独特な雰囲気が楽しめます。\n\
出力:\n\
    ポジティブ\n\
例3:\n\
入力:\n\
    猛暑日や暑い時期は注意が必要です。。\n\
出力:\n\
    ネガティブ\n\
例4:\n\
入力:\n\
    水遊びができる場所がたくさんある夏です。\n\
出力:\n\
    ポジティブ\n\
それでは次の入力に対して同様に出力を行ってください\n\
\n\
入力:\n\
{review}"
        prompts = []
        refined_sentences = []
        batch_size = 200
        for i in tqdm(range(0, len(df_general), batch_size)):
            prompts = []
            for j in range(batch_size):
                if i+j>=len(df_general):continue
                prompts.append(prompt.format(review=df_general.loc[i+j, 'topics']))
            outputs = self.llm.generate(prompts)
            print(outputs)
            refined_sentences+=outputs
            
        df_general['posneg_pp'] = refined_sentences
        return df_general
    
    def refine_sentence(self, df_simple: pd.DataFrame):
        '''
        df_simpleにおけるtopicsの文章を洗練させる
        '''
        prompt = "次の文章から内容の重複や繰り返しを取り除き，日本語として簡潔で自然な文章に変えてください.また，多数の文章からなる場合は読みやすくなるように要約してください．次は例です\n\
例1:\n\
入力:\n\
    「優しく丁寧に教えていただき、十分に楽しむことができました。楽しみ方のコツも教えていただき、大満足です。」<|im_end|>\n\
    丁寧な説明と指導のおかげで、のんびりと楽しい時間を過ごすことができました。<|im_end|>\n\
出力:\n\
    丁寧で優しい説明と指導によりのんびりと楽しめました．コツも教えていただき大満足です\n\
例2:\n\
入力:\n\
現代美術館では、斬新なデザインや趣のある建物が展示されており、モダンな空気が演出されています。モダンな作品が豊富に展示されており、こじんまりとした雰囲気も漂っています。<|im_end|>\n\
 芸術祭でたくさんの人が訪れて、様々な作品が展示されていました。展示品はアート性が高く、五感を刺激する一風変わった作品が多かった。また、ここでしか見られない資料や作品も展示されていました。展示作品は豊富で、これからの作品の数々が楽しみになるような場所でした。<|im_end|>\n\
出力:\n\
    現代美術館では、斬新なデザインや趣のある建物が展示されており、モダンな空気が演出されています．芸術祭では多くの人が訪れ，モダンな作品やアート性の高い，五感を刺激する一風変わった作品など豊富な作品が展示されています．\n\
それでは次の入力に対して同様に出力を行ってください\n\
\n\
入力:\n\
{review}"
        prompts = []
        refined_sentences = []
        batch_size = 200
        for i in tqdm(range(0, len(df_simple), batch_size)):
            prompts = []
            for j in range(batch_size):
                if i+j>=len(df_simple):continue
                prompts.append(prompt.format(review=df_simple.loc[i+j, 'combined_topic']))
            outputs = self.llm.generate(prompts)
            print(outputs)
            refined_sentences+=outputs
            
        df_simple['refined_topics'] = refined_sentences
        return df_simple
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_topic_path', type=str, )
    parser.add_argument('--group_path', type=str, )
    parser.add_argument('--save_path', type=str, )
    args = parser.parse_args()
    
    pp = PostProcessor()

    df_general = pd.read_csv('/home/yamanishi/project/airport/src/data/topic_summary/topic_summary_incontext_kagawa_luke_all_general_merged.csv',
                             names=['index', 'spot', 'posneg', 'cluster', 'topics']).reset_index()

    df_general = pp.negative_check(df_general)
    print(df_general)
    df_general.to_csv('/home/yamanishi/project/airport/src/data/topic_summary/topic_summary_incontext_kagawa_luke_all_general_merged.csv')
