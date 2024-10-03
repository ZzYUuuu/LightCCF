import torch
import torch.nn.functional as F
import numpy as np
def get_bpr_loss(user_embed, pos_embed, neg_embed):

    pos_scores = torch.sum(torch.mul(user_embed, pos_embed), dim=1)
    neg_scores = torch.sum(torch.mul(user_embed, neg_embed), dim=1)

    loss = -torch.log(torch.sigmoid(pos_scores-neg_scores) + 10e-8)

    return torch.mean(loss)

def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])
    return reg_loss

def get_InfoNCE_loss(embedding1, embedding2, temperature):
    embedding1 = torch.nn.functional.normalize(embedding1)
    embedding2 = torch.nn.functional.normalize(embedding2)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1))
    total_score = torch.exp(total_score / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / total_score + 10e-6)
    return torch.mean(cl_loss)


def get_neighbor_aggregate_loss(embedding1, embedding2, tau):
    embedding1 = torch.nn.functional.normalize(embedding1)
    embedding2 = torch.nn.functional.normalize(embedding2)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / tau)

    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1)) + torch.matmul(embedding1, embedding1.transpose(0, 1))
    total_score = torch.exp(total_score / tau).sum(dim=1)

    na_loss = -torch.log(pos_score / total_score + 10e-6)
    return torch.mean(na_loss)


    return torch.mean(cl_loss)
   

