import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

from scipy.sparse import coo_matrix
import scipy.sparse as sp
from torch.autograd import grad


class DomainDecomposer(nn.Module):
    def __init__(self, input_dim, shared_dim, private_dim):
        super(DomainDecomposer, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, shared_dim),
            nn.ReLU()
        )
        self.private_source_mlp = nn.Sequential(
            nn.Linear(input_dim, private_dim),
            nn.ReLU()
        )
        self.private_target_mlp = nn.Sequential(
            nn.Linear(input_dim, private_dim),
            nn.ReLU()
        )
    
    def forward(self, source_u_feats, target_u_feats):
        combined = torch.cat([source_u_feats, target_u_feats], dim=1)
        shared_embedding = self.shared_mlp(combined)
        source_private = self.private_source_mlp(source_u_feats)  
        target_private = self.private_target_mlp(target_u_feats)  
        return shared_embedding, source_private, target_private

class Proposed(nn.Module):
    def __init__(self,dropout=0,**params):
        super(Proposed, self).__init__()
        self.batch_size = params["batch_size"]
        self.p = 0
        self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1
        self.beta = 1
        self.gamma = params['gamma']
        
        self.device = params['device']
        self.dropout = dropout
        self.weight = nn.Parameter(torch.randn(self.batch_size))
        self.user_num = params['num_users']
        self.source_item_num = params['source_num_items']
        self.target_item_num = params['target_num_items']
        self.embed_id_dim = params['embed_id_dim']
        self.text_embed_dim = params['text_embed_dim']
        self.visual_embed_dim = params['visual_embed_dim']
        self.review_embed_dim = params['review_embed_dim']
        self.source_ui_inter_lists_train = params['source_ui_inter_lists_train']
        self.target_ui_inter_lists_train = params['target_ui_inter_lists_train']
        self.source_text_feat = params['source_text_feat']
        self.source_visual_feat = params['source_visual_feat']
        self.source_review_feat = params['source_review_feat']
        self.target_text_feat = params['target_text_feat']
        self.target_visual_feat = params['target_visual_feat']
        self.target_review_feat = params['target_review_feat']
        self.n_layers = params['n_layers']
        self.wo = params['wo']
        self.input_dim = self.embed_id_dim 
        self.shared_dim = self.embed_id_dim 
        self.private_dim = self.embed_id_dim 
        self.source_n_nodes = self.user_num + self.source_item_num
        self.target_n_nodes = self.user_num + self.target_item_num

        self.source_u_emb = nn.Embedding(params['num_users'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.source_u_emb.weight)
        self.target_u_emb = nn.Embedding(params['num_users'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.target_u_emb.weight)

        self.source_v_emb = nn.Embedding(params['source_num_items'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.source_v_emb.weight)
        self.target_v_emb = nn.Embedding(params['target_num_items'], params['embed_id_dim']).to(self.device)
        nn.init.xavier_uniform_(self.target_v_emb.weight)

        self.source_text_feat_emb = nn.Embedding(self.source_item_num, self.text_embed_dim).to(self.device)
        self.source_text_feat_emb.weight.data.copy_(self.source_text_feat)
        self.source_text_feat_emb.weight.requires_grad = False

        self.source_visual_feat_emb = nn.Embedding(self.source_item_num, self.visual_embed_dim).to(self.device)
        self.source_visual_feat_emb.weight.data.copy_(self.source_visual_feat)
        self.source_visual_feat_emb.weight.requires_grad = False

        self.source_review_feat_emb = nn.Embedding(self.user_num, self.review_embed_dim).to(self.device)
        self.source_review_feat_emb.weight.data.copy_(self.source_review_feat)
        self.source_review_feat_emb.weight.requires_grad = False

        self.target_text_feat_emb = nn.Embedding(self.target_item_num, self.text_embed_dim).to(self.device)
        self.target_text_feat_emb.weight.data.copy_(self.target_text_feat)
        self.target_text_feat_emb.weight.requires_grad = False

        self.target_visual_feat_emb = nn.Embedding(self.target_item_num, self.visual_embed_dim).to(self.device)
        self.target_visual_feat_emb.weight.data.copy_(self.target_visual_feat)
        self.target_visual_feat_emb.weight.requires_grad = False

        self.target_review_feat_emb = nn.Embedding(self.user_num, self.review_embed_dim).to(self.device)
        self.target_review_feat_emb.weight.data.copy_(self.target_review_feat)
        self.target_review_feat_emb.weight.requires_grad = False

        self.source_v_feat_layer = nn.Linear(self.embed_id_dim,self.embed_id_dim).to(self.device)
        self.target_v_feat_layer = nn.Linear(self.embed_id_dim, self.embed_id_dim).to(self.device)

        self.source_mat = self.create_sparse_matrix(params['source_train_data'],self.user_num,self.source_item_num)
        self.target_mat = self.create_sparse_matrix(params['target_train_data'],self.user_num,self.target_item_num)
        self.source_norm_adj = self.get_norm_adj_mat(self.source_mat.astype(np.float32),self.user_num,self.source_item_num,self.source_n_nodes).to(self.device)
        self.target_norm_adj = self.get_norm_adj_mat(self.target_mat.astype(np.float32),self.user_num,self.target_item_num,self.target_n_nodes).to(self.device)
        self.source_u_g_embeddings, self.source_v_g_embeddings = self.get_user_item_id_emb(self.source_u_emb,self.source_v_emb,self.user_num,self.source_item_num,self.source_norm_adj)
        self.target_u_g_embeddings, self.target_v_g_embeddings = self.get_user_item_id_emb(self.target_u_emb,
                                                                                           self.target_v_emb,
                                                                                           self.user_num,
                                                                                           self.target_item_num,
                                                                                           self.target_norm_adj)
        
        self.source_text_feat_layer = nn.Linear(self.text_embed_dim,self.embed_id_dim).to(self.device)
        self.target_text_feat_layer = nn.Linear(self.text_embed_dim,self.embed_id_dim).to(self.device)
        self.source_visual_feat_layer = nn.Linear(self.visual_embed_dim,self.embed_id_dim).to(self.device)
        self.target_visual_feat_layer = nn.Linear(self.visual_embed_dim,self.embed_id_dim).to(self.device)
        self.source_review_feat_layer = nn.Linear(self.review_embed_dim, self.embed_id_dim).to(self.device)
        self.target_review_feat_layer = nn.Linear(self.review_embed_dim, self.embed_id_dim).to(self.device)

        self.shared_mlp = nn.Sequential(
            nn.Linear(self.embed_id_dim * 2, self.embed_id_dim),
            nn.ReLU()
        )
        
        self.private_source_mlp = nn.Sequential(
            nn.Linear(self.embed_id_dim, self.embed_id_dim),
            nn.ReLU()
        )
        
        self.private_target_mlp = nn.Sequential(
            nn.Linear(self.embed_id_dim, self.embed_id_dim),
            nn.ReLU()
        )

        self.source_concat_layer = nn.Linear(self.embed_id_dim * 2, self.embed_id_dim).to(self.device)
        self.target_concat_layer = nn.Linear(self.embed_id_dim * 2, self.embed_id_dim).to(self.device)

        self.decomposer = DomainDecomposer(self.embed_id_dim, self.embed_id_dim, self.embed_id_dim)

        self.source_pred_1 = nn.Linear(self.embed_id_dim*2,self.embed_id_dim).to(self.device)
        self.source_norm_1 = nn.BatchNorm1d(self.embed_id_dim).to(self.device)
        self.source_pred_2 = nn.Linear(self.embed_id_dim, self.embed_id_dim//2).to(self.device)
        self.source_norm_2 = nn.BatchNorm1d(self.embed_id_dim//2).to(self.device)
        self.source_pred_3 = nn.Linear(self.embed_id_dim // 2, 1).to(self.device)

        self.target_pred_1 = nn.Linear(self.embed_id_dim * 2, self.embed_id_dim).to(self.device)
        self.target_norm_1 = nn.BatchNorm1d(self.embed_id_dim).to(self.device)
        self.target_pred_2 = nn.Linear(self.embed_id_dim, self.embed_id_dim // 2).to(self.device)
        self.target_norm_2 = nn.BatchNorm1d(self.embed_id_dim // 2).to(self.device)
        self.target_pred_3 = nn.Linear(self.embed_id_dim // 2, 1).to(self.device)

        self.criterion = nn.BCELoss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def create_sparse_matrix(self, df_feat, user_num,item_num,form='coo', value_field=None):
        src = df_feat['userID'].values
        tgt = df_feat['itemID'].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(user_num, item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def get_norm_adj_mat(self, interaction_matrix,user_num,item_num,n_nodes):
        A = sp.dok_matrix((user_num + item_num,
                           user_num + item_num), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + user_num),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + user_num, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((n_nodes, n_nodes)))

    def get_user_item_id_emb(self,u_emb,v_emb,user_num,item_num,norm_adj):

        h = v_emb.weight

        ego_embeddings = torch.cat((u_emb.weight, v_emb.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [user_num, item_num], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def generate_user_emb(self,nodes_u,u_g_embeddings,v_g_embeddings,ui_inter_lists,domain_flag):
        embed_matrix = torch.empty(len(nodes_u), self.embed_id_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes_u)):
            e_u = u_g_embeddings[int(nodes_u[i])]
            ui_inters = ui_inter_lists[int(nodes_u[i])]
            id_feat = v_g_embeddings[ui_inters]
            if domain_flag == 'source':
                text_feat = self.source_text_feat_emb(torch.tensor(ui_inters).to(self.device))
                text_feat = F.relu(self.source_text_feat_layer(text_feat))
                visual_feat = self.source_visual_feat_emb(torch.tensor(ui_inters).to(self.device))
                visual_feat = F.relu(self.source_visual_feat_layer(visual_feat))
            else:
                text_feat = self.target_text_feat_emb(torch.tensor(ui_inters).to(self.device))
                text_feat = F.relu(self.target_text_feat_layer(text_feat))
                visual_feat = self.target_visual_feat_emb(torch.tensor(ui_inters).to(self.device))
                visual_feat = F.relu(self.target_visual_feat_layer(visual_feat))
            e_v = id_feat+text_feat+visual_feat
            att_w = self.att.forward(e_v, e_u, len(ui_inters))
            att_history = torch.mm(e_v.t(), att_w).t()
            embed_matrix[i] = att_history
        return embed_matrix  
    
    def forward(self, source_nodes_u, source_nodes_v,target_nodes_u,target_nodes_v):
        source_u_review_feats = self.source_review_feat_emb(source_nodes_u)
        source_u_review_feats = F.relu(self.source_review_feat_layer(source_u_review_feats))
        target_u_review_feats = self.target_review_feat_emb(target_nodes_u)
        target_u_review_feats = F.relu(self.target_review_feat_layer(target_u_review_feats))
        source_u_id_feats = self.source_u_g_embeddings[source_nodes_u]
        target_u_id_feats = self.target_u_g_embeddings[target_nodes_u]
        source_v_id_feats = self.source_v_g_embeddings[source_nodes_v]
        target_v_id_feats = self.target_v_g_embeddings[target_nodes_v]
        if self.wo == "id":
            source_u_feats = source_u_review_feats
            target_u_feats = target_u_review_feats
        elif self.wo == "review":
            source_u_feats = source_u_id_feats
            target_u_feats = target_u_id_feats
        else:
            source_u_feats = source_u_id_feats + source_u_review_feats
            target_u_feats = target_u_id_feats + target_u_review_feats
        source_v_text_feat = self.source_text_feat_emb(source_nodes_v)
        source_v_text_feat = F.relu(self.source_text_feat_layer(source_v_text_feat))
        source_v_visual_feat = self.source_visual_feat_emb(source_nodes_v)
        source_v_visual_feat = F.relu(self.source_visual_feat_layer(source_v_visual_feat))
        target_v_text_feat = self.target_text_feat_emb(target_nodes_v)
        target_v_text_feat = F.relu(self.target_text_feat_layer(target_v_text_feat))
        target_v_visual_feat = self.target_visual_feat_emb(target_nodes_v)
        target_v_visual_feat = F.relu(self.target_visual_feat_layer(target_v_visual_feat))
        
        if self.wo == "text":
            source_v_feats = source_v_id_feats + source_v_visual_feat
            target_v_feats = target_v_id_feats + target_v_visual_feat
        elif self.wo == "visual":
            source_v_feats = source_v_id_feats + source_v_text_feat 
            target_v_feats = target_v_id_feats + target_v_text_feat 
        elif self.wo == "id":
            source_v_feats = source_v_visual_feat + source_v_text_feat 
            target_v_feats = target_v_visual_feat + target_v_text_feat 
        else:
            source_v_feats = source_v_id_feats + source_v_visual_feat + source_v_text_feat 
            target_v_feats = target_v_id_feats + target_v_visual_feat + target_v_text_feat

        shared_embedding, source_private, target_private = self.decomposer(source_u_feats, target_u_feats)
        if self.wo == "com":
            source_u_feats = source_private
            target_u_feats = target_private
        elif self.wo == "spe":
            source_u_feats = shared_embedding
            target_u_feats = shared_embedding
        else:
            source_u_feats = F.relu(
                self.source_concat_layer(torch.cat([shared_embedding, source_private], dim=1)))
            target_u_feats = F.relu(
                self.target_concat_layer(torch.cat([shared_embedding, target_private], dim=1)))

        source_v_feats = F.relu(self.source_v_feat_layer(source_v_feats))
        source_pred = torch.concat([source_u_feats,source_v_feats],dim=1)
        source_pred = self.source_norm_1(F.relu(self.source_pred_1(source_pred)))
        source_pred = self.source_norm_2(F.relu(self.source_pred_2(source_pred)))
        source_pred = F.sigmoid(self.source_pred_3(source_pred))

        target_v_feats = F.relu(self.target_v_feat_layer(target_v_feats))
        target_pred = torch.concat([target_u_feats, target_v_feats],dim=1)
        target_pred = self.target_norm_1(F.relu(self.target_pred_1(target_pred)))
        target_pred = self.target_norm_2(F.relu(self.target_pred_2(target_pred)))
        target_pred = F.sigmoid(self.target_pred_3(target_pred))

        return source_pred.squeeze(), target_pred.squeeze(), shared_embedding, source_private, target_private
    
    def loss(self, source_pred,target_pred,source_label,target_label,shared_embedding, source_private, target_private):
        source_pred_loss = self.criterion(source_pred,source_label.to(torch.float32))
        target_pred_loss = self.criterion(target_pred,target_label.to(torch.float32))
        loss_rating = source_pred_loss + target_pred_loss
        
        loss_cos_shared_source = cosine_loss(shared_embedding, source_private)
        loss_cos_shared_target = cosine_loss(shared_embedding, target_private)
        loss_cos_source_target = cosine_loss(source_private, target_private)

        loss_vec = loss_cos_shared_source + loss_cos_shared_target + loss_cos_source_target
        total_loss = loss_rating + self.gamma * loss_vec
        return total_loss, loss_vec, source_pred_loss, target_pred_loss

def cosine_loss(shared, private):
    cos_sim = F.cosine_similarity(shared, private, dim=1)
    loss = torch.mean(cos_sim ** 2)
    return loss