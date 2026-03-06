import numpy as np
import torch


def extract_axis_1(data, indices):
    rows = [data[i, indices[i], :] for i in range(data.shape[0])]
    return torch.stack(rows, dim=0).unsqueeze(1)


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    padded = list(itemlist)
    padded.extend([pad_item] * (length - len(itemlist)))
    return padded


def calculate_hit(sorted_list, topk, true_items, rewards, r_click, total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
    for i, top_k in enumerate(topk):
        rec_list = sorted_list[:, -top_k:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = top_k - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
