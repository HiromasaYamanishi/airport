import pandas as pd

from requests_html import HTMLSession
from bs4 import BeautifulSoup
import random
import time
from tqdm import tqdm
import numpy as np
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from pyquery import PyQuery as pq
import requests
import re
import time
import random
import pickle
import os
import urllib
import csv
from tqdm import tqdm
from urllib.parse import urljoin
from urllib.request import urlretrieve
import difflib




def collect_hotel_html():
    df_hotel = pd.read_csv('/home/yamanishi/project/airport/src/data/02_Travel_HotelMaster.csv', encoding='cp932')

    session = HTMLSession()
    for hotel_id, hotel_name in tqdm(zip(df_hotel['施設ID'], df_hotel['施設名'])):
        info_url = f"https://travel.rakuten.co.jp/HOTEL/{hotel_id}/{hotel_id}.html"
        map_url = f'https://travel.rakuten.co.jp/HOTEL/{hotel_id}/rtmap.html'
        gallery_url = "https://travel.rakuten.co.jp/HOTEL/90/gallery.html"
        review_url = "https://travel.rakuten.co.jp/HOTEL/90/gallery.html"
        r_info = session.get(info_url)
        
        hotel_name = hotel_name.replace('/', '')
        with open(f'/home/yamanishi/project/airport/src/data/hotel/html/{hotel_name}_info.html', 'w') as f:
            f.write(r_info.html.html)
        time.sleep(random.uniform(0.35, 0.65))
        r_map = session.get(map_url)
        with open(f'/home/yamanishi/project/airport/src/data/hotel/html/{hotel_name}_map.html', 'w') as f:
            f.write(r_map.html.html)
            
        time.sleep(random.uniform(0.35, 0.65))
        
def make_hotel_info_df():
    df = pd.read_csv('/home/yamanishi/project/airport/src/data/02_Travel_HotelMaster.csv', encoding='cp932')
    prefectures, latitudes, longitudes, review_counts, hotel_names, hotel_ids = [], [], [], [], [], []
    for hotel_id, hotel_name in tqdm(zip(df['施設ID'], df['施設名'])):
        hotel_name = hotel_name.replace('/', '')
        with open(f'/home/yamanishi/project/airport/src/data/hotel/html/{hotel_name}_map.html', 'r') as f:
            html = f.read()
            d = BeautifulSoup(html, 'lxml')
            try:
                prefecture = d.find('select', {'id': 'pref'}).text.strip()
            except AttributeError:
                prefecture = None
            try:
                latitude = d.find('input', {'name': 'latitude'})['value']
            except TypeError:
                latitude = None
            try:
                longitude = d.find('input', {'name': 'longitude'})['value']
            except TypeError:
                longitude = None
            #print(prefecture, latitude, longitude)
        with open(f'/home/yamanishi/project/airport/src/data/hotel/html/{hotel_name}_info.html', 'r') as f:
            html_info = f.read()
            #print(html_info)
            d = BeautifulSoup(html_info, 'lxml')
            try:
                element = d.find('a', {'property': 'reviewCount'})
                review_count = element.get('content')
            except AttributeError:
                review_count = None
        hotel_names.append(hotel_name)
        hotel_ids.append(hotel_id)
        prefectures.append(prefecture)
        latitudes.append(latitude)
        longitudes.append(longitude)
        review_counts.append(review_count)
            
            
    df_out = pd.DataFrame.from_dict({'hotel_id': hotel_ids, 'hotel_name': hotel_names, 'prefecture': prefectures, 
                                'review_count': review_counts, 'latitude': latitudes, 'longitude': longitudes,
                                })
    
    df_out.to_csv('/home/yamanishi/project/airport/src/data/hotel_info.csv')
    
class JalanPath:
    '''
    保存するパスを定義
    '''
    def __init__(self):
        self.data_dir = '/home/yamanishi/project/trip_recommend/data'
        self.data_jalan_dir = os.path.join(self.data_dir, 'jalan')

        self.data_jalan_spot_dir = os.path.join(self.data_jalan_dir, 'spot')
        self.spot_html_dir = os.path.join(self.data_jalan_spot_dir, 'html')
        self.spot_all_csv_path = os.path.join(self.data_jalan_spot_dir, 'spot_all.csv')
        self.spot_all_again_csv_path = os.path.join(self.data_jalan_spot_dir, 'spot_all_again.csv')
        self.spot_info_all_csv_path = os.path.join(self.data_jalan_spot_dir, 'spot_info_all.csv')
        self.spot_info_all_again_csv_path = os.path.join(self.data_jalan_spot_dir, 'spot_info_all_again.csv')
        self.spot_review_dir = os.path.join(self.data_jalan_spot_dir, 'review')

        self.data_jalan_food_dir = os.path.join(self.data_jalan_dir, 'food')
        self.food_html_dir = os.path.join(self.data_jalan_food_dir, 'html')
        self.food_all_csv_path = os.path.join(self.data_jalan_food_dir, 'food_all.csv')
        self.food_info_all_csv_path = os.path.join(self.data_jalan_food_dir, 'food_info_all.csv')
        self.food_review_dir = os.path.join(self.data_jalan_food_dir, 'review')
        self.session = HTMLSession() 
        self.prefectures = """
                            北海道
                            青森県
                            岩手県
                            宮城県
                            秋田県
                            山形県
                            福島県
                            茨城県
                            栃木県
                            群馬県
                            埼玉県
                            千葉県
                            東京都
                            神奈川県
                            新潟県
                            富山県
                            石川県
                            福井県
                            山梨県
                            長野県
                            岐阜県
                            静岡県
                            愛知県
                            三重県
                            滋賀県
                            京都府
                            大阪府
                            兵庫県
                            奈良県
                            和歌山県
                            鳥取県
                            島根県
                            岡山県
                            広島県
                            山口県
                            徳島県
                            香川県
                            愛媛県
                            高知県
                            福岡県
                            佐賀県
                            長崎県
                            熊本県
                            大分県
                            宮崎県
                            鹿児島県
                            沖縄県
                            """
        self.prefectures_list = re.findall(r'(\w+)\n', self.prefectures)
        self.itenary_all_path = os.path.join(self.data_jalan_dir, 'itenary_all.npy')


class GetInfoAll(JalanPath):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('../data/df_hotel_kagawa.csv')

    def save_html(self, r, spot_name):
        with open(os.path.join(self.data_jalan_spot_dir, f'{spot_name}.txt'),'wb') as f:
            f.write(r.content)
        return
    
    def get_review_count_rate(self, soup):
        review_element = soup.find('a', {'property': 'reviewCount'})
        if review_element is None:
            return 0, 0
        review_count_text = review_element.find('em').text
        review_count = int(review_count_text.replace(',', ''))
        rating_element = soup.find('strong', {'class': 'rating'})
        # テキスト部分を抽出
        review_rating = float(rating_element.text)
        return review_count, review_rating
                

    def read_html(self, spot_name, html_dir):
        spot_name = spot_name.replace('/','')
        if not os.path.exists(os.path.join(html_dir, f'{spot_name}.html')):
            return False
        with open(os.path.join(html_dir, f'{spot_name}.html'), 'r') as f:
            html = f.read()
        return html

    def get_area(self, soup, type='kankou'):
        '''
        都市、町、地域を取得
        type: kankouかgourment
        '''
        try:
            town = soup.select(f'a[href^="//www.jalan.net/{type}/tow"]')[0].text
        except IndexError:
            town = None
        
        try:
            city = soup.select(f'a[href^="//www.jalan.net/{type}/cit"]')[1].text
        except IndexError:
            city = None
        try:
            area = soup.select(f'div.dropdownCurrent a[href^="//www.jalan.net/{type}/"]')
            area = [a.text for a in area]
            area = area[1]
        except IndexError:
            area = None

        return area, city, town
    
    def get_hotel_name_web(self, soup):
        meta_tag = soup.find('meta', {'property': 'name'})
        hotel_name = meta_tag['content'] if meta_tag else None
        return hotel_name

    def get_jenre(self,soup, type='kankou'):
        '''
        場所のカテゴリを取得
        '''
        jenres = soup.select(f'dl.c-genre a[href^="//www.jalan.net/{type}/g"]')
        jenres = [j.text for j in jenres]
        return jenres

    def get_location(self,soup):
        '''
        場所の緯度経度を取得
        '''
        map = lat = soup.select('td#detailMap > div.detailMap-canvas')
        lat, lng = None, None
        if len(map)>0:
            lat = map[0].get('data-lat')
            lng = map[0].get('data-lng')
        return lat, lng

    def get_review_rate_count(self,soup):
        '''
        場所のレビュー数とレート数を取得
        '''
        point, count= None, None
        points = soup.select('div.detailHeader-ratingArea .reviewPoint')
        if len(points)>0:
            if points[0].text=='-.-':
                point=0
            else:
                point = float(points[0].text)

        counts = soup.select('div.detailHeader-ratingArea .reviewCount')
        if len(counts)>0:
            pattern = r'(\d*,*\d+)件'
            m=re.search(pattern, counts[0].text)
            if m is not None:
                count = int(m.group(1).replace(',',''))

        return point, count
    
    def get_true_review_count(self, soup):
        paging_area = soup.find('div', {'class': 'pagingArea'})
        total_count_element = paging_area.find('em')
        total_count = int(total_count_element.text)
        return total_count
    
    def get_user_names(self, soup):
        users = soup.find_all('span', class_=['hotel', 'user'])
        return [u.text.strip() for u in users]
    
    def get_review_sentences(self, soup):
        comment_sentence = soup.find_all('p', class_='commentSentence')
        reviews = [c.text.strip() for c in comment_sentence]
        return reviews
    
    def get_purpose_accompony_time(self, soup):
        dl_elements = soup.find_all('dl', {'class': 'commentPurpose'})
        dd_elements = [dl_element.find_all('dd') for dl_element in dl_elements]
        extracted_texts = [[dd.text for dd in dd_element] for dd_element in dd_elements]
        purposes = [e[0] if len(e)>0 else None for e in extracted_texts]
        accompanies = [e[1] if len(e)>1 else None for e in extracted_texts]
        visit_times = [e[2] if len(e)>2 else None for e in extracted_texts]
        return purposes, accompanies, visit_times
    
    def convert_to_float(self, s):
        try:
            s = float(s)
            return s
        except ValueError:
            return 0
    
    def get_ratings(self, soup):
        rating_elements = soup.find_all('span', class_=re.compile('rate rate*'))
        ratings = [element.text for element in rating_elements]
        ratings = [self.convert_to_float(element) for element in ratings]
        return ratings
    
    def get_plan_rooms(self, soup):
        rooms = []
        plans = []
        dt_elements = soup.find_all('div', {'class': 'commentNote'})
        for dt in dt_elements:
            plan_dt = dt.find('dt', text='ご利用の宿泊プラン')
            if plan_dt is None:
                plans.append(None)
            else:
                plan_dd = plan_dt.find_next_sibling('dd')
                plan_text = plan_dd.get_text(strip=True)
                plans.append(plan_text)
            room_dt = dt.find('dt', text='ご利用のお部屋')
            if room_dt is None:
                rooms.append(None)
            else:
                room_dd = room_dt.find_next_sibling('dd')
                room_text = room_dd.get_text(strip=True)
                rooms.append(room_text)
                
        return rooms, plans
    
    def find_closest_hotel(self, hotel_name, hotel_list):
        """
        与えられたホテル名に最も近い名前をリストから探す関数

        Args:
        hotel_name (str): 探すホテル名
        hotel_list (list): ホテル名のリスト

        Returns:
        str: リスト内で最も近いホテル名
        """
        closest_match = difflib.get_close_matches(hotel_name, hotel_list, n=1, cutoff=0.2)
        return closest_match[0] if closest_match else None
        
    def get_review(self, start=0):
        '''
        レビューを全て取得
        '''
        hotel_df = pd.read_csv('../data/hotel_info.csv')
        #hotel_df = hotel_df[hotel_df['prefecture']=='香川県'].reset_index()
        review_df_path = os.path.join('../data/hotel/review', 'review_all_period.csv')
        if not os.path.exists(review_df_path):
            review_df = pd.DataFrame(columns =['hotel_name','pref','review','rating','accompany',
                    'purpose','name','visit_time', 'plan', 'room']) 
            review_df.to_csv(review_df_path)
        review_df = pd.read_csv(review_df_path)

        df_review_tmp = []
        for i in range(start, len(hotel_df)):
            hotel_id, hotel_name, prefecture = hotel_df.loc[i, 'hotel_id'], hotel_df.loc[i, 'hotel_name'], hotel_df.loc[i, 'prefecture']
            print(hotel_id,hotel_name,)
            review_url = f'https://review.travel.rakuten.co.jp/hotel/voice/{hotel_id}/?f_time=&f_keyword=&f_age=0&f_sex=0&f_mem1=0&f_mem2=0&f_mem3=0&f_mem4=0&f_mem5=0&f_teikei=&f_version=2&f_static=1&f_point=0&f_sort=0&f_jrdp=0&f_next=0'
            session = HTMLSession()
            r = session.get(review_url)
            d = BeautifulSoup(r.html.html, 'lxml')
            review_count, review_stars = self.get_review_count_rate(d)
            print(review_count)
            #review_count = self.get_true_review_count(d)
            for page_num in range((review_count-1)//20+1):
                print(page_num)
                sleep_time = random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
                kuchikomi_page_url = f'https://review.travel.rakuten.co.jp/hotel/voice/{hotel_id}/?f_time=&f_keyword=&f_age=0&f_sex=0&f_mem1=0&f_mem2=0&f_mem3=0&f_mem4=0&f_mem5=0&f_teikei=&f_version=2&f_static=1&f_point=0&f_sort=0&f_jrdp=0&f_next={page_num*20}'
                session = HTMLSession()
                r = session.get(kuchikomi_page_url)
                soup = BeautifulSoup(r.html.html, 'lxml')
                users = self.get_user_names(soup)
                #if page_num==0:
                hotel_name_in_page=self.get_hotel_name_web(soup)
                print(hotel_name_in_page)
                hotel_name_in_page = self.find_closest_hotel(hotel_name_in_page, users)
                print(hotel_name_in_page)
                purposes, accompanies, visit_times = self.get_purpose_accompony_time(soup)
                if len(users)==20 and len(purposes)==20:
                    is_user = [True for _ in range(20)]
                else:
                    is_user = [u!=hotel_name_in_page for u in users]
                review_sentences = self.get_review_sentences(soup)
                user_names = [users[k] for k in range(len(review_sentences)) if is_user[k]]
                #print(user_names)
                user_review_sentences = [review_sentences[k] for k in range(len(review_sentences)) if is_user[k]]
                ratings = self.get_ratings(soup)[-len(user_names):]
                if len(ratings)<len(user_names):
                    for _ in range(len(user_names)-len(ratings)):
                        ratings.append(0)
                if len(ratings)>len(user_names):
                    ratings = ratings[:len(user_names)]
                prefs = [prefecture for _ in range(len(user_names))]
                hotel_names = [hotel_name for _ in range(len(user_names))]
                plans, rooms = self.get_plan_rooms(soup)
                print(len(hotel_names), len(prefs), len(user_review_sentences),
                      len(ratings), len(accompanies), len(purposes), len(user_names),
                      len(visit_times), len(plans), len(rooms))
                d={'hotel_name':hotel_names,
                            'pref' : prefs,
                            'review':user_review_sentences,
                            'rating':ratings,
                            'accompany': accompanies,
                            'purpose':purposes,
                            'name':user_names,
                            'visit_time':visit_times,
                            'plan': plans,
                            'room': rooms}

                df_hotel_tmp = pd.DataFrame(d)
                df_review_tmp.append(df_hotel_tmp)
                if len(df_hotel_tmp)==0:
                    break
            start+=1
            if start%5==0:
                if len(df_review_tmp)>0:
                    df_review_tmp = pd.concat(df_review_tmp)
                    df_review_tmp.to_csv(review_df_path, mode='a', header=False)
                    df_review_tmp=[]

        df_review_tmp = pd.concat(df_review_tmp)
        df_review_tmp.to_csv(review_df_path, mode='a', header=False)



if __name__=='__main__':
    gia = GetInfoAll()
    gia.get_review(start=5)
    #make_hotel_info_df()
    