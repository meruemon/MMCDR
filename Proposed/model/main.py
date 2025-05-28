import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pandas as pd
import numpy as np
from loguru import logger
import os
import random as rd
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings("ignore")

from utils.evaluation import calculate_hr_ndcg
from utils.para_parser import parse
from utils.path_params import phone_sport, movie_music

import multiprocessing
from functools import partial
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from Proposed import Proposed


def visualize_final_embeddings(model, source_nodes_u, source_nodes_v,
                               target_nodes_u, target_nodes_v, gamma,
                               sample_num=500):
    model.eval()
    with torch.no_grad():
        _, _, shared_emb, source_private, target_private = model(
            source_nodes_u.to(model.device),
            source_nodes_v.to(model.device),
            target_nodes_u.to(model.device),
            target_nodes_v.to(model.device)
        )

    shared_emb = shared_emb.detach().cpu().numpy()[:sample_num]
    source_private = source_private.detach().cpu().numpy()[:sample_num]
    target_private = target_private.detach().cpu().numpy()[:sample_num]

    all_emb = np.vstack([shared_emb, source_private, target_private])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_emb_2d = tsne.fit_transform(all_emb)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_emb_2d[:sample_num, 0], all_emb_2d[:sample_num, 1],
                label='Shared', color='red', alpha=0.7)
    plt.scatter(all_emb_2d[sample_num:2*sample_num, 0], all_emb_2d[sample_num:2*sample_num, 1],
                label='Source Private', color='blue', alpha=0.7)
    plt.scatter(all_emb_2d[2*sample_num:, 0], all_emb_2d[2*sample_num:, 1],
                label='Target Private', color='green', alpha=0.7)
    plt.savefig(f"TSNE_{gamma}_Movie.pdf")
    plt.show()


def generate_user_inter_lists(df):
    ui_interaction = {}
    for x in df.groupby(by='userID'):
        ui_interaction[x[0]] = list(x[1]['itemID'])
    return ui_interaction


def load_data_with_validation(path_inter, path_text_feat, path_review_feat, path_visual_feat):
    df = pd.read_csv(path_inter)
    df['split'] = 'train'
    test_idx = df.groupby('userID').sample(n=1, random_state=42).index
    df.loc[test_idx, 'split'] = 'test'
    
    remaining = df[df['split'] == 'train']
    val_idx = remaining.groupby('userID').sample(n=1, random_state=42).index
    df.loc[val_idx, 'split'] = 'val'
    
    train_data = df[df['split'] == 'train'][['userID', 'itemID', 'rating']]
    val_data   = df[df['split'] == 'val'][['userID', 'itemID', 'rating']]
    test_data  = df[df['split'] == 'test'][['userID', 'itemID', 'rating']]
    
    num_users = len(df['userID'].unique().tolist())
    num_items = len(df['itemID'].unique().tolist())
    
    ui_inter_lists = generate_user_inter_lists(df)
    ui_inter_lists_train = generate_user_inter_lists(train_data)
    ui_inter_lists_val   = generate_user_inter_lists(val_data)
    ui_inter_lists_test  = generate_user_inter_lists(test_data)
    
    with open(path_text_feat, 'rb') as f:
        text_feat = torch.from_numpy(np.load(f)).to(device)
    with open(path_visual_feat, 'rb') as f:
        visual_feat = torch.from_numpy(np.load(f)).to(device)
    with open(path_review_feat, 'rb') as f:
        review_feat = torch.from_numpy(np.load(f)).to(device)
    
    return (num_users, num_items, ui_inter_lists, text_feat, review_feat, visual_feat,
            train_data, val_data, test_data, df, ui_inter_lists_train, ui_inter_lists_val, ui_inter_lists_test)

def generate_train_batch_for_all_overlap(source_user_inters, source_user_inters_valid, source_user_inters_test, source_num_items,
                                         target_user_inters, target_user_inters_valid, target_user_inters_test, target_num_items,
                                         batch_size):
    t_source = []
    t_target = []
    for b in range(batch_size // 2):
        u = rd.sample(list(source_user_inters.keys()), 1)[0]
        i_source = rd.sample(source_user_inters[u], 1)[0]
        i_target = rd.sample(target_user_inters[u], 1)[0]
        t_source.append([u, i_source, 1])
        t_target.append([u, i_target, 1])
        
        pos_src = set(source_user_inters[u]) \
                | set(source_user_inters_valid[u]) \
                | set(source_user_inters_test[u])
        pos_tgt = set(target_user_inters[u]) \
                | set(target_user_inters_valid[u]) \
                | set(target_user_inters_test[u])

        j_source = rd.randint(0, source_num_items-1)
        while j_source in pos_src:
            j_source = rd.randint(0, source_num_items-1)

        j_target = rd.randint(0, target_num_items-1)
        while j_target in pos_tgt:
            j_target = rd.randint(0, target_num_items-1)
        
        t_source.append([u, j_source, 0])
        t_target.append([u, j_target, 0])
        
    return np.asarray(t_source), np.asarray(t_target)

def _sample_negatives(pos_set: set, num_items: int, n: int):
    negs = set()
    while len(negs) < n:
        k = rd.randint(0, num_items - 1)     
        if k not in pos_set:
            negs.add(k)
    return list(negs)

def generate_valid_batch_for_all_overlap(
    src_train, src_valid, src_test, src_num_items,
    tgt_train, tgt_valid, tgt_test, tgt_num_items,
    n_neg
):
    src_pos, src_neg = [], []
    tgt_pos, tgt_neg = [], []

    for u in src_train.keys():          
        v_item = src_valid[u][0]   
        t_item = tgt_valid[u][0]

        src_pos.append([u, v_item])
        tgt_pos.append([u, t_item])

        pos_src = set(src_train[u]) | set(src_valid[u]) | set(src_test[u])
        pos_tgt = set(tgt_train[u]) | set(tgt_valid[u]) | set(tgt_test[u])

        src_neg.append(_sample_negatives(pos_src, src_num_items, n_neg))
        tgt_neg.append(_sample_negatives(pos_tgt, tgt_num_items, n_neg))

    return src_pos, src_neg, tgt_pos, tgt_neg

def generate_test_batch_for_all_overlap(
    src_train, src_valid, src_test, src_num_items,
    tgt_train, tgt_valid, tgt_test, tgt_num_items,
    n_neg
):
    src_pos, src_neg = [], []
    tgt_pos, tgt_neg = [], []

    for u in src_train.keys():
        t_item_src = src_test[u][0]
        t_item_tgt = tgt_test[u][0]

        src_pos.append([u, t_item_src])
        tgt_pos.append([u, t_item_tgt])

        pos_src = set(src_train[u]) | set(src_valid[u]) | set(src_test[u])
        pos_tgt = set(tgt_train[u]) | set(tgt_valid[u]) | set(tgt_test[u])

        src_neg.append(_sample_negatives(pos_src, src_num_items, n_neg))
        tgt_neg.append(_sample_negatives(pos_tgt, tgt_num_items, n_neg))

    return src_pos, src_neg, tgt_pos, tgt_neg


def train(model, device, optimizer, source_ui_inter_lists, source_ui_inter_lists_valid, source_ui_inter_lists_test, source_num_items,
          target_ui_inter_lists, target_ui_inter_lists_valid, target_ui_inter_lists_test, target_num_items, batch_size, bar_length):
    logger.info('訓練開始')
    model.train()
    total_loss = []
    total_loss_vec = []
    total_loss_a = []
    total_loss_b = []

    for idx in tqdm(range(bar_length), desc=f"Epoch {epoch}"):
        uij_source, uij_target = generate_train_batch_for_all_overlap(source_ui_inter_lists, source_ui_inter_lists_valid, source_ui_inter_lists_test,
                                                                      source_num_items,
                                                                      target_ui_inter_lists, target_ui_inter_lists_valid, target_ui_inter_lists_test,
                                                                      target_num_items, batch_size)
        source_batch_users = uij_source[:, 0]
        source_batch_items = uij_source[:, 1]
        source_batch_r     = uij_source[:, 2]
        target_batch_users = uij_target[:, 0]
        target_batch_items = uij_target[:, 1]
        target_batch_r     = uij_target[:, 2]
        
        optimizer.zero_grad()

        source_pred, target_pred, shared_embedding, source_private, target_private = model.forward(
            torch.tensor(source_batch_users).to(device),
            torch.tensor(source_batch_items).to(device),
            torch.tensor(target_batch_users).to(device),
            torch.tensor(target_batch_items).to(device)
        )
        loss, loss_vec, loss_a, loss_b = model.loss(source_pred, target_pred,
                          torch.tensor(source_batch_r, dtype=torch.float).to(device),
                          torch.tensor(target_batch_r, dtype=torch.float).to(device),
                          shared_embedding, source_private, target_private)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        total_loss.append(loss.item())
        total_loss_vec.append(loss_vec.item())
        total_loss_a.append(loss_a.item())
        total_loss_b.append(loss_b.item())
    return total_loss, total_loss_vec, total_loss_a, total_loss_b

def sample_user_item_pairs(ui_dict, num_users, num_samples=500):
    user_ids = []
    item_ids = []
    count = 0
    for uid in range(num_users):
        items = ui_dict.get(uid, [])
        if len(items) == 0:
            continue
        for iid in items:
            user_ids.append(uid)
            item_ids.append(iid)
            count += 1
            if count >= num_samples:
                return torch.LongTensor(user_ids), torch.LongTensor(item_ids)
    return torch.LongTensor(user_ids), torch.LongTensor(item_ids)


def valid(model, device, top_k, epoch, gamma, source_user_inters, source_user_inters_valid, source_user_inters_test, source_num_items,
         target_user_inters, target_user_inters_valid, target_user_inters_test, target_num_items, test_neg_num):
    logger.info('評価開始')
    model.eval()
    with torch.no_grad():
        logger.info('テストまたは検証用の正例・負例サンプル生成')
        source_test_pos, source_test_neg, target_test_pos, target_test_neg = generate_valid_batch_for_all_overlap(
            source_user_inters, source_user_inters_valid, source_user_inters_test, source_num_items,
            target_user_inters, target_user_inters_valid, target_user_inters_test, target_num_items, test_neg_num)
        
        (source_hr, source_ndcg, source_precision, source_recall, source_mrr,
         target_hr, target_ndcg, target_precision, target_recall, target_mrr,
         source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1,
         target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1,
         source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5,
         target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5) = calculate_hr_ndcg(
            model, source_test_pos, source_test_neg, target_test_pos, target_test_neg, top_k, device, epoch)
    
    # source_nodes_u, source_nodes_v = sample_user_item_pairs(source_user_inters_test, model.user_num, num_samples=3000)
    # target_nodes_u, target_nodes_v = sample_user_item_pairs(target_user_inters_test, model.user_num, num_samples=3000)
        
    # visualize_final_embeddings(
    #     model,
    #     source_nodes_u,
    #     source_nodes_v,
    #     target_nodes_u,
    #     target_nodes_v,
    #     gamma,
    #     sample_num=3000
    # )
        

    return (source_hr, source_ndcg, source_precision, source_recall, source_mrr,
            target_hr, target_ndcg, target_precision, target_recall, target_mrr,
            source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1,
            target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1,
            source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5,
            target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5)


def test(model, device, top_k, epoch, gamma, source_user_inters, source_user_inters_valid, source_user_inters_test, source_num_items,
         target_user_inters, target_user_inters_valid, target_user_inters_test, target_num_items, test_neg_num):
    logger.info('評価開始')
    model.eval()
    with torch.no_grad():
        logger.info('テストまたは検証用の正例・負例サンプル生成')
        source_test_pos, source_test_neg, target_test_pos, target_test_neg = generate_test_batch_for_all_overlap(
            source_user_inters, source_user_inters_valid, source_user_inters_test, source_num_items,
            target_user_inters, target_user_inters_valid, target_user_inters_test, target_num_items, test_neg_num)
        
        source_nodes_u, source_nodes_v = sample_user_item_pairs(source_user_inters_test, model.user_num, num_samples=3000)
        target_nodes_u, target_nodes_v = sample_user_item_pairs(target_user_inters_test, model.user_num, num_samples=3000)
        
        visualize_final_embeddings(
            model,
            source_nodes_u,
            source_nodes_v,
            target_nodes_u,
            target_nodes_v,
            gamma,
            sample_num=3000
        )
        
        (source_hr, source_ndcg, source_precision, source_recall, source_mrr,
         target_hr, target_ndcg, target_precision, target_recall, target_mrr,
         source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1,
         target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1,
         source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5,
         target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5) = calculate_hr_ndcg(
            model, source_test_pos, source_test_neg, target_test_pos, target_test_neg, top_k, device, epoch)
    return (source_hr, source_ndcg, source_precision, source_recall, source_mrr,
            target_hr, target_ndcg, target_precision, target_recall, target_mrr,
            source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1,
            target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1,
            source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5,
            target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5)

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    args = parse()
    if args.datasets == "phone_sport":
        dataset_name = phone_sport
    elif args.datasets == "movie_music":
        dataset_name = movie_music
    else:
        dataset_name = movie_music
    print(args.datasets)

    logger.info('ソースデータ読み込み')
    (source_num_users, source_num_items, source_ui_inter_lists, source_text_feat, source_review_feat, source_visual_feat,
     source_train_data, source_val_data, source_test_data, source_df,
     source_ui_inter_lists_train, source_ui_inter_lists_val, source_ui_inter_lists_test) = load_data_with_validation(
         dataset_name['source_path_inter'], dataset_name['source_path_text_feat'],
         dataset_name['source_path_review_feat'], dataset_name['source_path_visual_feat'])
    
    logger.info('ターゲットデータ読み込み')
    (target_num_users, target_num_items, target_ui_inter_lists, target_text_feat, target_review_feat, target_visual_feat,
     target_train_data, target_val_data, target_test_data, target_df,
     target_ui_inter_lists_train, target_ui_inter_lists_val, target_ui_inter_lists_test) = load_data_with_validation(
         dataset_name['target_path_inter'], dataset_name['target_path_text_feat'],
         dataset_name['target_path_review_feat'], dataset_name['target_path_visual_feat'])
    
    logger.info(f'source ユーザー数: {source_num_users}, source アイテム数: {source_num_items}, 訓練データ数: {len(source_train_data)}')
    logger.info(f'target ユーザー数: {target_num_users}, target アイテム数: {target_num_items}, 訓練データ数: {len(target_train_data)}')
    logger.info(f'datase: {args.datasets}, emb_dim: {args.embed_id_dim},  lr: {args.lr}, batch_size: {args.batch_size}, epochs: {args.epochs}, gamma: {args.gamma}, wo: {args.wo}')
    
    params = {
        'num_users': source_num_users,
        'source_num_items': source_num_items,
        'source_train_data': source_train_data,
        'source_ui_inter_lists_train': source_ui_inter_lists_train,
        'source_text_feat': source_text_feat,
        'source_visual_feat': source_visual_feat,
        'source_review_feat': source_review_feat,
        'target_num_items': target_num_items,
        'target_train_data': target_train_data,
        'target_ui_inter_lists_train': target_ui_inter_lists_train,
        'target_text_feat': target_text_feat,
        'target_visual_feat': target_visual_feat,
        'target_review_feat': target_review_feat,
        'embed_id_dim': args.embed_id_dim,
        'text_embed_dim': args.text_embed_dim,
        'visual_embed_dim': args.visual_embed_dim,
        'review_embed_dim': args.review_embed_dim,
        'field': args.field,
        'n_layers': args.n_layers,
        'batch_size': args.batch_size,
        'device': device,
        'gpu_id': args.gpu_id,
        'gamma': args.gamma,
        'wo': args.wo
    }
    
    torch.cuda.set_device(params['gpu_id'])
    
    dcdr = Proposed(**params, args=args)
    dcdr.to(device)
    optimizer = torch.optim.Adam(dcdr.parameters(), lr=args.lr)
    
    logger.info('訓練および評価開始')
    bar_length = dataset_name['bar_length']
    best_model_path = dataset_name['save_model_path'].replace('.pth', '_best.pth')
    last_model_path = dataset_name['save_model_path'].replace('.pth', '_last.pth')

    source_best_hr, source_best_ndcg, target_best_hr, target_best_ndcg = 0.0, 0.0, 0.0, 0.0
    endure_count = 0
    
    for epoch in range(1, args.epochs + 1):
        total_loss, loss_vec, loss_a, loss_b = train(dcdr, device, optimizer,
                           source_ui_inter_lists_train, source_ui_inter_lists_val, source_ui_inter_lists_test, source_num_items,
                           target_ui_inter_lists_train, target_ui_inter_lists_val, target_ui_inter_lists_test, target_num_items,
                           args.batch_size, bar_length)
        
        loss_vec_total =  sum(loss_vec) / len(loss_vec)
        total_loss_a = sum(loss_a) / len(loss_a)
        total_loss_b = sum(loss_b) / len(loss_b)
        epoch_loss = sum(total_loss) / len(total_loss)
        logger.info('epoch {} 訓練損失: {}'.format(epoch, epoch_loss))
        logger.info('epoch {} コサイン類似度損失: {}'.format(epoch, loss_vec_total))
        logger.info('epoch {} 訓練_A損失: {}'.format(epoch, total_loss_a))
        logger.info('epoch {} 訓練_B損失: {}'.format(epoch, total_loss_b))

        (source_hr, source_ndcg, source_precision, source_recall, source_mrr,
         target_hr, target_ndcg, target_precision, target_recall, target_mrr,
         source_hr_1, source_ndcg_1, source_precision_1, source_recall_1, source_mrr_1,
         target_hr_1, target_ndcg_1, target_precision_1, target_recall_1, target_mrr_1,
         source_hr_5, source_ndcg_5, source_precision_5, source_recall_5, source_mrr_5,
         target_hr_5, target_ndcg_5, target_precision_5, target_recall_5, target_mrr_5) = valid(
            dcdr, device, args.top_k, epoch, args.gamma, 
            source_ui_inter_lists_train, source_ui_inter_lists_val, source_ui_inter_lists_test, source_num_items,
            target_ui_inter_lists_train, target_ui_inter_lists_val, target_ui_inter_lists_test, target_num_items,
            args.test_neg_num)
        
        logger.info('[%d] s_hr_1: %.4f, s_NDCG_1: %.4f, s_mrr_1:%.4f, s_prec_1: %.4f,'
                    's_hr_5: %.4f, s_NDCG_5: %.4f, s_mrr_5:%.4f, s_prec_5: %.4f,'
                    's_hr_10: %.4f, s_NDCG_10: %.4f, s_mrr_10:%.4f, s_prec_10: %.4f,' %
                    (epoch, source_hr_1,source_ndcg_1,source_mrr_1,source_precision_1,
                     source_hr_5,source_ndcg_5,source_mrr_5,source_precision_5,
                     source_hr,source_ndcg,source_mrr,source_precision))
        logger.info('[%d] t_hr_1: %.4f, t_NDCG_1: %.4f, t_mrr_1:%.4f, t_prec_1: %.4f,'
                    't_hr_5: %.4f, t_NDCG_5: %.4f, t_mrr_5:%.4f, t_prec_5: %.4f,'
                    't_hr_10: %.4f, t_NDCG_10: %.4f, t_mrr_10:%.4f, t_prec_10: %.4f,' %
                    (epoch, target_hr_1, target_ndcg_1, target_mrr_1, target_precision_1,
                     target_hr_5, target_ndcg_5, target_mrr_5, target_precision_5,
                     target_hr, target_ndcg, target_mrr, target_precision))

        if target_hr > target_best_hr:
            source_best_hr = source_hr
            source_best_ndcg = source_ndcg
            target_best_hr = target_hr
            target_best_ndcg = target_ndcg
            torch.save(dcdr.state_dict(), best_model_path)
            endure_count = 0
            logger.info("epoch {} で新たなベストモデルを保存 (検証指標: {})".format(epoch, target_hr))
        else:
            endure_count += 1
        

        torch.save(dcdr.state_dict(), last_model_path)
    
    logger.info("訓練完了。最良の検証モデルは {} に保存されました。".format(best_model_path))
    
    dcdr.load_state_dict(torch.load(best_model_path))
    dcdr.eval()
    (val_source_hr, val_source_ndcg, val_source_precision, val_source_recall, val_source_mrr,
     val_target_hr, val_target_ndcg, val_target_precision, val_target_recall, val_target_mrr,
     val_source_hr_1, val_source_ndcg_1, val_source_precision_1, val_source_recall_1, val_source_mrr_1,
     val_target_hr_1, val_target_ndcg_1, val_target_precision_1, val_target_recall_1, val_target_mrr_1,
     val_source_hr_5, val_source_ndcg_5, val_source_precision_5, val_source_recall_5, val_source_mrr_5,
     val_target_hr_5, val_target_ndcg_5, val_target_precision_5, val_target_recall_5, val_target_mrr_5) = valid(
        dcdr, device, args.top_k, epoch, args.gamma, 
        source_ui_inter_lists_train, source_ui_inter_lists_val, source_ui_inter_lists_test, source_num_items,
        target_ui_inter_lists_train, target_ui_inter_lists_val, target_ui_inter_lists_test, target_num_items,
        args.test_neg_num)
    

    csv_file_path_val_best = 'results_log_Proposed_val_best.csv'
    row_dict_val_best = {
        'datasets': args.datasets,
        'val_source_hr_1': val_source_hr_1,
        'val_source_NDCG_1': val_source_ndcg_1,
        'val_source_mrr_1': val_source_mrr_1,
        'val_source_prec_1': val_source_precision_1,
        'val_source_hr_5': val_source_hr_5,
        'val_source_NDCG_5': val_source_ndcg_5,
        'val_source_mrr_5': val_source_mrr_5,
        'val_source_hr_10': val_source_hr,
        'val_source_NDCG_10': val_source_ndcg,
        'val_source_mrr_10': val_source_mrr,
        'val_target_hr_1': val_target_hr_1,
        'val_target_NDCG_1': val_target_ndcg_1,
        'val_target_mrr_1': val_target_mrr_1,
        'val_target_hr_5': val_target_hr_5,
        'val_target_NDCG_5': val_target_ndcg_5,
        'val_target_mrr_5': val_target_mrr_5,
        'val_target_hr_10': val_target_hr,
        'val_target_NDCG_10': val_target_ndcg,
        'val_target_mrr_10': val_target_mrr
    }
    df_val_best = pd.DataFrame([row_dict_val_best])
    file_exists = os.path.isfile(csv_file_path_val_best)
    df_val_best.to_csv(csv_file_path_val_best, mode='a', index=False, header=not file_exists)
    logger.info("検証用【ベストモデル】の結果をCSVに出力しました。ファイルパス: %s", csv_file_path_val_best)
    
    dcdr.load_state_dict(torch.load(last_model_path))
    dcdr.eval()
    (val_source_hr, val_source_ndcg, val_source_precision, val_source_recall, val_source_mrr,
     val_target_hr, val_target_ndcg, val_target_precision, val_target_recall, val_target_mrr,
     val_source_hr_1, val_source_ndcg_1, val_source_precision_1, val_source_recall_1, val_source_mrr_1,
     val_target_hr_1, val_target_ndcg_1, val_target_precision_1, val_target_recall_1, val_target_mrr_1,
     val_source_hr_5, val_source_ndcg_5, val_source_precision_5, val_source_recall_5, val_source_mrr_5,
     val_target_hr_5, val_target_ndcg_5, val_target_precision_5, val_target_recall_5, val_target_mrr_5) = test(
        dcdr, device, args.top_k, epoch, args.gamma, 
        source_ui_inter_lists_train, source_ui_inter_lists_val, source_ui_inter_lists_test, source_num_items,
        target_ui_inter_lists_train, target_ui_inter_lists_val, target_ui_inter_lists_test, target_num_items,
        args.test_neg_num)
    
    csv_file_path_val_last = 'results_log_Proposed_val_last.csv'
    row_dict_val_last = {
        'datasets': args.datasets,
        'val_source_hr_1': val_source_hr_1,
        'val_source_NDCG_1': val_source_ndcg_1,
        'val_source_mrr_1': val_source_mrr_1,
        'val_source_prec_1': val_source_precision_1,
        'val_source_hr_5': val_source_hr_5,
        'val_source_NDCG_5': val_source_ndcg_5,
        'val_source_mrr_5': val_source_mrr_5,
        'val_source_hr_10': val_source_hr,
        'val_source_NDCG_10': val_source_ndcg,
        'val_source_mrr_10': val_source_mrr,
        'val_target_hr_1': val_target_hr_1,
        'val_target_NDCG_1': val_target_ndcg_1,
        'val_target_mrr_1': val_target_mrr_1,
        'val_target_hr_5': val_target_hr_5,
        'val_target_NDCG_5': val_target_ndcg_5,
        'val_target_mrr_5': val_target_mrr_5,
        'val_target_hr_10': val_target_hr,
        'val_target_NDCG_10': val_target_ndcg,
        'val_target_mrr_10': val_target_mrr
    }
    df_val_last = pd.DataFrame([row_dict_val_last])
    file_exists = os.path.isfile(csv_file_path_val_last)
    df_val_last.to_csv(csv_file_path_val_last, mode='a', index=False, header=not file_exists)
    logger.info("検証用【最終モデル】の結果をCSVに出力しました。ファイルパス: %s", csv_file_path_val_last)
    
    dcdr.load_state_dict(torch.load(last_model_path))
    dcdr.eval()
    (test_source_hr, test_source_ndcg, test_source_precision, test_source_recall, test_source_mrr,
     test_target_hr, test_target_ndcg, test_target_precision, test_target_recall, test_target_mrr,
     test_source_hr_1, test_source_ndcg_1, test_source_precision_1, test_source_recall_1, test_source_mrr_1,
     test_target_hr_1, test_target_ndcg_1, test_target_precision_1, test_target_recall_1, test_target_mrr_1,
     test_source_hr_5, test_source_ndcg_5, test_source_precision_5, test_source_recall_5, test_source_mrr_5,
     test_target_hr_5, test_target_ndcg_5, test_target_precision_5, test_target_recall_5, test_target_mrr_5) = test(
        dcdr, device, args.top_k, epoch, args.gamma,
        source_ui_inter_lists_train, source_ui_inter_lists_val, source_ui_inter_lists_test, source_num_items,
        target_ui_inter_lists_train, target_ui_inter_lists_val, target_ui_inter_lists_test, target_num_items,
        args.test_neg_num)
    
    if args.datasets == "phone_sport":
        csv_file_path_test = 'results_log_Proposed_test_sport_phone.csv'
    else: 
        csv_file_path_test = 'results_log_Proposed_test_movie_music.csv'
    row_dict_test = {
    'datasets': args.datasets,
    'gamma': args.gamma,
    'wo': args.wo,


    'test_source_hr_1': test_source_hr_1,
    'test_source_NDCG_1': test_source_ndcg_1,
    'test_source_mrr_1': test_source_mrr_1,
    'test_source_precision_1': test_source_precision_1,
    'test_source_recall_1': test_source_recall_1,


    'test_source_hr_5': test_source_hr_5,
    'test_source_NDCG_5': test_source_ndcg_5,
    'test_source_mrr_5': test_source_mrr_5,
    'test_source_precision_5': test_source_precision_5,
    'test_source_recall_5': test_source_recall_5,


    'test_source_hr_10': test_source_hr,
    'test_source_NDCG_10': test_source_ndcg,
    'test_source_mrr_10': test_source_mrr,
    'test_source_precision_10': test_source_precision,
    'test_source_recall_10': test_source_recall,


    'test_target_hr_1': test_target_hr_1,
    'test_target_NDCG_1': test_target_ndcg_1,
    'test_target_mrr_1': test_target_mrr_1,
    'test_target_precision_1': test_target_precision_1,
    'test_target_recall_1': test_target_recall_1,


    'test_target_hr_5': test_target_hr_5,
    'test_target_NDCG_5': test_target_ndcg_5,
    'test_target_mrr_5': test_target_mrr_5,
    'test_target_precision_5': test_target_precision_5,
    'test_target_recall_5': test_target_recall_5,

    'test_target_hr_10': test_target_hr,
    'test_target_NDCG_10': test_target_ndcg,
    'test_target_mrr_10': test_target_mrr,
    'test_target_precision_10': test_target_precision,
    'test_target_recall_10': test_target_recall,
}
    df_test = pd.DataFrame([row_dict_test])
    file_exists = os.path.isfile(csv_file_path_test)
    df_test.to_csv(csv_file_path_test, mode='a', index=False, header=not file_exists)
    logger.info("テスト用の結果をCSVに出力しました。ファイルパス: %s", csv_file_path_test)
    print("CSVファイルに結果を追記しました。")
