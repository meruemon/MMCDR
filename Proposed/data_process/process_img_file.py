import os
import array
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import ast
from tqdm import tqdm  

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
vgg19.eval()
vgg19.to(device)


truncated_classifier = nn.Sequential(*list(vgg19.classifier.children())[:2])

truncated_classifier.to(device)

feature_extractor = nn.Sequential(
    *list(vgg19.features.children()),
    vgg19.avgpool,
    nn.Flatten(),
    *list(truncated_classifier.children())
)
feature_extractor.eval()
feature_extractor.to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_vgg19_feature_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to download image: {url} - {e}")
        return None
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img_t = transform(img).unsqueeze(0)  
    img_t = img_t.to(device)
    with torch.no_grad():
        feat = feature_extractor(img_t)
    return feat.squeeze().cpu().numpy()

def extract_first_url(raw):
    """
    画像情報の文字列（例: "[{'360w': 'https://...jpg', '480w': '...', ...}]"）
    をパースして、最初の辞書の最初の値（順序どおり）を返す。
    """
    try:
        data = ast.literal_eval(raw)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            first_value = next(iter(data[0].values()))
            return first_value
    except Exception as e:
        print("Error parsing image info:", e)
    return None

def write_image_features_to_b(csv_path, output_b_path):
    df = pd.read_csv(csv_path, header=None)
    df = df.dropna(subset=[0, 1])
    df = df.drop_duplicates(subset=[0])
    
    total_items = len(df)
    with open(output_b_path, 'wb') as f:
        for idx, row in tqdm(df.iterrows(), total=total_items, desc="Writing features", ascii=True):
            asin = str(row[0])
            asin_fixed = asin.ljust(10)[:10]
            f.write(asin_fixed.encode('UTF-8'))
            
            raw_img_info = row[1]
            url = extract_first_url(raw_img_info)
            if not url:
                print(f"Failed to extract URL for asin {asin}. Using empty URL.")
                url = ""
            feat = extract_vgg19_feature_from_url(url)
            if feat is None:
                feat = np.zeros(4096, dtype='float32')
            else:
                feat = feat.astype('float32')
            a = array.array('f', feat)
            a.tofile(f)
            
    print("Finished writing .b file.")

if __name__ == '__main__':
    dataset = "phone_sport"
    if dataset == "phone_elec":
        csv_path = '../datasets/movie_music/music/meta_music_data.csv'
        output_b_path = '../datasets/movie_music/music/image_features_CDs_and_Vinyl.b'
        write_image_features_to_b(csv_path, output_b_path)
        print("Finished!")
    elif dataset == "movie_music":
        csv_path = '../datasets/movie_music/movie/meta_movie_data.csv'
        output_b_path = '../datasets/movie_music/movie/image_features_Movies_and_TV.b'
        write_image_features_to_b(csv_path, output_b_path)
        csv_path = '../datasets/movie_music/music/meta_music_data.csv'
        output_b_path = '../datasets/movie_music/music/image_features_CDs_and_Vinyl.b'
        write_image_features_to_b(csv_path, output_b_path)
        print("Finished!")
    elif dataset == "phone_sport":
        csv_path = '../datasets/phone_sport/sport/meta_sport_data.csv'
        output_b_path = '../datasets/phone_sport/sport/image_features_Sports_and_Outdoors.b'
        write_image_features_to_b(csv_path, output_b_path)
        csv_path = '../datasets/phone_sport/phone/meta_phone_data.csv'
        output_b_path = '../datasets/phone_sport/phone/image_features_Cell_Phones_and_Accessories.b'
        write_image_features_to_b(csv_path, output_b_path)
        print("Finished!")
    
