import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

def knn(query, data, k=10):
    assert data.shape[1] == query.shape[1]

    M = torch.cdist(query, data)
    v, ind = M.topk(k, largest=False)

    return v, ind[:, 0:min(k, data.shape[0])].to(torch.long)


def sample_knn_labels(query_embd, y_query, prior_embd, labels, k=10, n_class=10, weighted=False):
    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_indices = torch.randint(0, k + 1, (n_sample,))
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), sampled_indices]

    # compute the frequency of sampled labels
    neighbour_freq = torch.sum(neighbour_label_distribution, dim=1)

    # normalize max count as weight
    if weighted:
        weights = torch.sum(neighbour_freq * sampled_labels, dim=1) / torch.sum(neighbour_freq, dim=1)
    else:
        weights = 1 / n_sample * torch.ones([n_sample]).to(query_embd.device)

    return sampled_labels, torch.squeeze(weights)

def estimate_knn_labels(query_embd, y_query, prior_embd, labels, k=10, n_class=10, weighted=False, Hard=False):
    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # compute the frequency of sampled labels
    neighbour_freq = torch.sum(neighbour_label_distribution, dim=1)

    # normalize to get probability distribution
    sampled_labels = neighbour_freq / (k + 1)  # Normalize by k + 1 (k neighbours + itself)

    # compute weights
    if weighted:
        weights = torch.sum(neighbour_freq * sampled_labels, dim=1) / torch.sum(neighbour_freq, dim=1)
    else:
        weights = 1 / n_sample * torch.ones([n_sample]).to(query_embd.device)

    if Hard:
        sampled_labels = (sampled_labels > 0.5).float()

    return sampled_labels, torch.squeeze(weights)


def knn_labels(neighbours, indices, k=20, n_class=20):
    n_sample = len(indices)

    # compute the label of nearest neighbours
    neighbour_label_distribution = torch.tensor(neighbours[indices, :k+1]).to(torch.long)

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_indices = torch.randint(0, k + 1, (n_sample,))
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), sampled_indices]

    # compute the frequency of sampled labels
    neighbour_freq = torch.sum(neighbour_label_distribution, dim=1)

    # normalize max count as weight
    weights = torch.sum(neighbour_freq * sampled_labels, dim=1) / torch.sum(neighbour_freq, dim=1)

    return sampled_labels, torch.squeeze(weights)


def mean_knn_labels(query_embd, y_query, prior_embd, labels, k=100, n_class=10):
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # compute the mean of labels
    mean_labels = torch.mean(neighbour_label_distribution, dim=1)

    return mean_labels


def prepare_knn(labels, train_embed, save_dir, k=10):
    if os.path.exists(save_dir):
        neighbours = torch.tensor(np.load(save_dir))
        print(f'knn were computed before, loaded from: {save_dir}')
    else:
        neighbours = torch.zeros([train_embed.shape[0], k + 1, labels.shape[1]]).to(torch.float)
        for i in tqdm(range(int(train_embed.shape[0] / 1000) + 1), desc='Searching knn', ncols=100):
            start = i * 1000
            end = min((i + 1) * 1000, train_embed.shape[0])
            ebd = train_embed[start:end, :]
            _, neighbour_ind = knn(ebd, train_embed, k=k + 1)
            neighbours[start:end, :] = labels[neighbour_ind]
        np.save(save_dir, neighbours.numpy())

    return neighbours

def calculate_neighborhood_label_variance(embeddings, labels, k=10):
    n_samples = embeddings.shape[0]
    device = embeddings.device 
    
    _, neighbour_ind = knn(embeddings, embeddings, k=k)

    variance = []
    
    for i in tqdm(range(n_samples), desc="Computing neighborhood variances"):
        neighbor_labels = labels[neighbour_ind[i]]
        
        current_label = labels[i]
        
        current_label_repeated = torch.ones_like(neighbor_labels, device=device) * current_label
        
        diff = neighbor_labels - current_label_repeated
        squared_diff = diff ** 2
        label_variance = torch.mean(squared_diff.float())
        
        variance.append(label_variance.item())
    
    return torch.tensor(variance, dtype=torch.float32)

def select_clean_samples(labels, variance, ratio=0.4):
    sorted_indices = torch.argsort(variance)
    
    n_clean = int(len(variance) * ratio)
    clean_indices = sorted_indices[:n_clean]

    clean_labels = labels[sorted_indices[:n_clean]]
    
    return clean_indices, clean_labels

def calculate_co_occurrence_matrix(clean_labels, n_class):
    co_occurrence = torch.zeros((n_class, n_class))
    
    for sample_labels in clean_labels:
        non_zero_indices = torch.where(sample_labels == 1)[0]
        
        for i in range(len(non_zero_indices)):
            for j in range(i+1, len(non_zero_indices)):
                c1 = non_zero_indices[i]
                c2 = non_zero_indices[j]
                co_occurrence[c1][c2] += 1
                co_occurrence[c2][c1] += 1  
    
    # 计算概率
    row_sum = torch.sum(co_occurrence, dim=1)
    for i in range(n_class):
        co_occurrence[i] = co_occurrence[i] / (row_sum[i] + 1e-8)  
    
    return co_occurrence

def estimate_knn_labels_matrix(query_embd, y_query, prior_embd, labels, k=10, n_class=10, weighted=True, Hard=False, co_occurrence_matrix=None):
    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    neighbour_label_distribution = labels[neighbour_ind]
    neighbour_freq = torch.sum(neighbour_label_distribution, dim=1)

    sampled_labels = (neighbour_freq + 1) / (k + 1) 

    if weighted and co_occurrence_matrix is not None:
        weights = torch.ones([n_sample]).to(query_embd.device)

        for i in range(n_sample):
            sample_labels = (sampled_labels[i] > 0.5).float()

            co_occurrence_scores = []
            for c1 in range(n_class):
                if sample_labels[c1] == 1:
                    for c2 in range(c1 + 1, n_class):
                        if sample_labels[c2] == 1:
                            co_occurrence_prob = co_occurrence_matrix[c1, c2]
                            co_occurrence_scores.append(co_occurrence_prob)

            if co_occurrence_scores:
                avg_rev_co = torch.mean(torch.tensor(co_occurrence_scores))
                max_rev_co = torch.max(torch.tensor(co_occurrence_scores))
                weights[i] = max_rev_co
            else:
                weights[i] = 1.0

        if torch.sum(weights) != 0:
            min_weight = torch.min(weights)
            max_weight = torch.max(weights)
            denominator = max_weight - min_weight + 1e-8
            scaled_weights = ((weights - min_weight) / denominator) * (1 - 0.5) + 0.5
            weights = scaled_weights
        else:
            weights = torch.ones([n_sample]).to(query_embd.device)

    elif weighted:
        weights = torch.sum(neighbour_freq * sampled_labels, dim=1) / (torch.sum(neighbour_freq, dim=1) + 1)
        weights = torch.where(torch.isnan(weights), torch.ones(n_sample).to(query_embd.device) / n_sample, weights)
    else:
        weights = 1 / n_sample * torch.ones([n_sample]).to(query_embd.device)

    if Hard:
        sampled_labels = (sampled_labels > 0.5).float()

    return sampled_labels, weights