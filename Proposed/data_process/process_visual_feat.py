import array

import pandas as pd
import numpy as np
from loguru import logger


def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin_bytes = f.read(10)
        if len(asin_bytes) < 10:
            break
        asin = asin_bytes.decode('UTF-8').strip()
        feature_bytes = f.read(4096 * 4)
        if len(feature_bytes) < 4096 * 4:
            print("Incomplete record encountered. Ending reading.")
            break
        
        a = array.array('f')
        a.frombytes(feature_bytes)
        yield asin, a.tolist()
def process_visual_feat(source_path,source_image_path,to_path):
  logger.info('read csv')
  df = pd.read_csv(source_path)
  df.sort_values(by=['item_id'], inplace=True)
  item2id = df['item_id'].unique().tolist()
  img_data = readImageFeatures(source_image_path)
  feats = {}
  avg = []
  logger.info('start image data')
  for d in img_data:
    if d[0] in item2id:
      feats[d[0]] = d[1]
      avg.append(d[1])
  avg = np.array(avg).mean(0).tolist()

  ret = []
  non_no = []
  logger.info('start filter')
  for i in item2id:
    if i in feats:
      ret.append(feats[i])
    else:
      non_no.append(i)
      ret.append(avg)

  logger.info('# of items not in processed image features:', len(non_no))
  assert len(ret) == len(item2id)
  np.save(to_path, np.array(ret))
  logger.info('complete')

if __name__ == '__main__':
  dataset_name = "phone_sport"
  if dataset_name == "movie_music":
    process_visual_feat(source_path='../datasets/movie_music/movie/movie_inter.csv',
                        source_image_path='../datasets/movie_music/movie/image_features_Movies_and_TV.b',
                        to_path='../datasets/movie_music/movie/movie_visual_feat.npy')
    
    process_visual_feat(source_path='../datasets/movie_music/music/music_inter.csv',
                        source_image_path='../datasets/movie_music/music/image_features_CDs_and_Vinyl.b',
                        to_path='../datasets/movie_music/music/music_visual_feat.npy')
  elif dataset_name == "phone_sport":
    process_visual_feat(source_path='../datasets/phone_sport/phone/phone_inter.csv',
                        source_image_path='../datasets/phone_sport/phone/image_features_Cell_Phones_and_Accessories.b',
                        to_path='../datasets/phone_sport/phone/phone_visual_feat.npy')
    
    process_visual_feat(source_path='../datasets/phone_sport/sport/sport_inter.csv',
                        source_image_path='../datasets/phone_sport/sport/image_features_Sports_and_Outdoors.b',
                        to_path='../datasets/phone_sport/sport/sport_visual_feat.npy')