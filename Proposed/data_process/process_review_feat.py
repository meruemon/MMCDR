import gzip
import pandas as pd
from sentence_transformers	import SentenceTransformer
import numpy as np
from loguru import logger

def select_reviews(source_path,to_path,items,users):
    g = gzip.open(source_path, 'r')
    review_list = []
    i = 0
    for line in g:
        d = eval(line, {"true": True, "false": False, "null": None})
        if (d['parent_asin'] in items) and (d['user_id'] in users):
            if i % 10000 == 0:
                logger.info(i)
            i+=1
            review_list.append([d['user_id'], d['parent_asin'],d['text']])
    df = pd.DataFrame(review_list, columns=['user_id', 'item_id','review_text'])  
    df.to_csv(to_path, index=False, encoding='utf-8', errors='replace')

def concat_reviews(source_path,to_path):
    df = pd.read_csv(source_path)
    df['review_text'].fillna(' ', inplace=True)
    df_lists = []
    logger.info('start group by')
    for x in df.groupby(by='user_id'):
        each_df = pd.DataFrame({
            'user_id': [x[0]],
            'review_texts': [';'.join(x[1]['review_text'])]
        })
        df_lists.append(each_df)

    df = pd.concat(df_lists, axis=0)
    logger.info('start store')
    df.to_csv(to_path, index=False, encoding='utf-8', errors='replace')

def generate_review_emb(source_path,source_path_text,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['user_id'], inplace=True)
    orig_user_ids  = df['user_id'].unique().tolist()
    df_text = pd.read_csv(source_path_text)
    logger.info('process null value')
    df_text.sort_values(by=['user_id'], inplace=True)
    sentences = []
    lack_index = []
    for each_user in orig_user_ids:
        row = df_text[df_text['user_id']==each_user]
        if len(row) == 0:
            print(each_user)
            lack_index.append(orig_user_ids.index(each_user))
        else:
            sen = row['review_texts'].iloc[0]
            sen = sen.replace('\n', ' ')
            sentences.append(sen)

    logger.info('start transform')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    fill_list = [0] * 384
    for each_index in lack_index:
        sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done')

if __name__ == '__main__':
    dataset = "phone_sport"
    if dataset == "movie_music":
        df = pd.read_csv('../datasets/movie_music/movie/movie_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Movies_and_TV.jsonl.gz',
                    to_path='../datasets/movie_music/movie/movie_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/movie_music/movie/movie_reviews_orig.csv',
                    to_path='../datasets/movie_music/movie/movie_reviews.csv')
        generate_review_emb(source_path='../datasets/movie_music/movie/movie_inter.csv',
                            source_path_text='../datasets/movie_music/movie/movie_reviews.csv',
                            to_path='../datasets/movie_music/movie/movie_review_feat.npy')

        df = pd.read_csv('../datasets/movie_music/music/music_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/CDs_and_Vinyl.jsonl.gz',
                    to_path='../datasets/movie_music/music/music_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/movie_music/music/music_reviews_orig.csv',
                    to_path='../datasets/movie_music/music/music_reviews.csv')
        generate_review_emb(source_path='../datasets/movie_music/music/music_inter.csv',
                            source_path_text='../datasets/movie_music/music/music_reviews.csv',
                            to_path='../datasets/movie_music/music/music_review_feat.npy')
        
    elif dataset == "phone_sport":
        df = pd.read_csv('../datasets/phone_sport/phone/phone_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Cell_Phones_and_Accessories.jsonl.gz',
                    to_path='../datasets/phone_sport/phone/phone_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/phone_sport/phone/phone_reviews_orig.csv',
                    to_path='../datasets/phone_sport/phone/phone_reviews.csv')
        generate_review_emb(source_path='../datasets/phone_sport/phone/phone_inter.csv',
                            source_path_text='../datasets/phone_sport/phone/phone_reviews.csv',
                            to_path='../datasets/phone_sport/phone/phone_review_feat.npy')
        df = pd.read_csv('../datasets/phone_sport/sport/sport_inter.csv')
        items = df['item_id'].unique().tolist()
        users = df['user_id'].unique().tolist()
        select_reviews(source_path='../datasets/Sports_and_Outdoors.jsonl.gz',
                    to_path='../datasets/phone_sport/sport/sport_reviews_orig.csv', items=items, users=users)
        concat_reviews(source_path='../datasets/phone_sport/sport/sport_reviews_orig.csv',
                    to_path='../datasets/phone_sport/sport/sport_reviews.csv')
        generate_review_emb(source_path='../datasets/phone_sport/sport/sport_inter.csv',
                            source_path_text='../datasets/phone_sport/sport/sport_reviews.csv',
                            to_path='../datasets/phone_sport/sport/sport_review_feat.npy')
    else:
        print("datasetを指定してください")