import torch
from torch import nn
import utility.trainer as trainer
import utility.tools as tools
import utility.losses as losses


class LightCCF(nn.Module):
    def __init__(self, args, dataset, device):
        super(LightCCF, self).__init__()
        self.model_name = "LightCCF"
        self.dataset = dataset
        self.args = args
        self.device = device
        self.reg_lambda = float(self.args.reg_lambda)
        self.activation = nn.Sigmoid()
        self.ssl_lambda = float(self.args.ssl_lambda)
        self.tau = float(self.args.tau)
        self.encoder = self.args.encoder
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.args.embedding_size))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.args.embedding_size))
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.adj_mat = self.dataset.sparse_adjacency_matrix()
        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)


    def aggregate(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [embeddings]

        for layer in range(int(self.args.gcn_layer)):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return user_emb, item_emb

    def forward(self, user, positive, negative):
        if self.encoder == 'MF':
            all_user_gcn_embed, all_item_gcn_embed = self.user_embedding.weight, self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]
        positive_gcn_embed = all_item_gcn_embed[positive.long()]
        negative_gcn_embed = all_item_gcn_embed[negative.long()]

        user_embed = self.user_embedding(user)
        positive_embed = self.item_embedding(positive)
        negative_embed = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_gcn_embed, positive_gcn_embed, negative_gcn_embed)
        reg_loss = losses.get_reg_loss(user_embed, positive_embed, negative_embed) * self.reg_lambda

        na_loss = losses.get_neighbor_aggregate_loss(user_gcn_embed, positive_gcn_embed, self.tau) * self.ssl_lambda

        loss_list = [bpr_loss, reg_loss, na_loss]

        return loss_list

    def get_rating_for_test(self, user):
        if self.encoder == 'MF':
            all_user_gcn_embed, all_item_gcn_embed = self.user_embedding.weight, self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]

        rating = self.activation(torch.matmul(user_gcn_embed, all_item_gcn_embed.t()))

        return rating


class Trainer():
    def __init__(self, args, dataset, device, logger):
        self.model = LightCCF(args, dataset, device)
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logger

    def train(self):
        trainer.training(self.model, self.args, self.dataset, self.device, self.logger)