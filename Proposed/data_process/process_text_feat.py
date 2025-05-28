import pandas as pd
import gzip
from sentence_transformers	import SentenceTransformer
import numpy as np
import os
from loguru import logger
from tqdm import tqdm
import torch

def select_data(df,meta_path,to_path):
    items = df['item_id'].unique().tolist()
    with gzip.open(meta_path, 'rb') as g:
        for line in tqdm(g, desc="Processing lines", ascii=True):
            d = eval(line, {"true":True,"false":False,"null":None})
            if d['parent_asin'] in items:
                try:
                    item_id = d['parent_asin']
                except:
                    item_id = ''
                try:
                    im_url = d['images']
                except:
                    im_url = ''
                try:
                    description = d['description']
                except:
                    description = ''
                try:
                    categories = ','.join(d['categories'])
                except:
                    categories = ''
                try:
                    title = d['title']
                except:
                    title = ''
                each_line = {
                    'item_id':[item_id],
                    'im_url':[im_url],
                    'description':[description],
                    'categories':[categories],
                    'title':[title]
                }
                each_line_df = pd.DataFrame(each_line)
                each_line_df.to_csv(to_path,mode='a',header=False,index=False, encoding='utf-8', errors='replace')

        
def process_text_data(source_path,source_path_text,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['item_id'], inplace=True)
    orig_item_ids  = df['item_id'].unique().tolist()
    df_text = pd.read_csv(source_path_text)
    df_text.columns = ['item_id', 'im_url', 'description', 'categories', 'title']
    logger.info('process null value')
    df_text.sort_values(by=['item_id'], inplace=True)
    df_text['description'] = df_text['description'].fillna(" ")
    df_text['title'] = df_text['title'].fillna(" ")
    df_text['categories'] = df_text['categories'].fillna(" ")
    sentences = []
    lack_index = []
    for each_item in orig_item_ids:
        row = df_text[df_text['item_id']==each_item]
        if len(row) == 0:
            print(each_item)
            lack_index.append(orig_item_ids.index(each_item))
        else:
            sen = row['title'].iloc[0] + ' ' + row['categories'].iloc[0] + ' ' + row['description'].iloc[0]
            sen = sen.replace('\n', ' ')
            sentences.append(sen)

    logger.info('start transform')
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    sentence_embeddings = model.encode(sentences)
    fill_list = [0] * 384
    for each_index in lack_index:
        sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done!')

if __name__ == '__main__':
    dataset = "phone_sport"
    if dataset == "movie_music":

        df = pd.read_csv('../datasets/movie_music/movie/movie_inter.csv')
        select_data(df,meta_path='../datasets/meta_Movies_and_TV.jsonl.gz',
                    to_path='../datasets/movie_music/movie/meta_movie_data.csv')
        process_text_data(source_path='../datasets/movie_music/movie/movie_inter.csv',
                        source_path_text='../datasets/movie_music/movie/meta_movie_data.csv',
                        to_path='../datasets/movie_music/movie/movie_text_feat.npy')
        

        df = pd.read_csv('../datasets/movie_music/music/music_inter.csv')
        select_data(df,meta_path='../datasets/meta_CDs_and_Vinyl.jsonl.gz',
                    to_path='../datasets/movie_music/music/meta_music_data.csv')
        process_text_data(source_path='../datasets/movie_music/music/music_inter.csv',
                        source_path_text='../datasets/movie_music/music/meta_music_data.csv',
                        to_path='../datasets/movie_music/music/music_text_feat.npy')
    elif dataset == "phone_sport":

        df = pd.read_csv('../datasets/phone_sport/phone/phone_inter.csv')
        select_data(df,meta_path='../datasets/meta_Cell_Phones_and_Accessories.jsonl.gz',
                    to_path='../datasets/phone_sport/phone/meta_phone_data.csv')
        process_text_data(source_path='../datasets/phone_sport/phone/phone_inter.csv',
                        source_path_text='../datasets/phone_sport/phone/meta_phone_data.csv',
                        to_path='../datasets/phone_sport/phone/phone_text_feat.npy')
        

        df = pd.read_csv('../datasets/phone_sport/sport/sport_inter.csv')
        select_data(df,meta_path='../datasets/meta_Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/phone_sport/sport/meta_sport_data.csv')
        process_text_data(source_path='../datasets/phone_sport/sport/sport_inter.csv',
                        source_path_text='../datasets/phone_sport/sport/meta_sport_data.csv',
                        to_path='../datasets/phone_sport/sport/sport_text_feat.npy')
    else:
        print("データセットを指定してください")