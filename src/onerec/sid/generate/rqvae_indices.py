import argparse
import collections
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from onerec.sid.datasets import EmbDataset
from onerec.sid.models.rqvae import RQVAE


def check_collision(indices_as_strings):
    return len(indices_as_strings) == len(set(indices_as_strings.tolist()))


def get_indices_count(indices_as_strings):
    counts = collections.defaultdict(int)
    for value in indices_as_strings:
        counts[value] += 1
    return counts


def get_collision_item(indices_as_strings):
    index2items = {}
    for item_id, index in enumerate(indices_as_strings):
        index2items.setdefault(index, []).append(item_id)
    return [items for items in index2items.values() if len(items) > 1]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SID indices from a trained RQ-VAE checkpoint")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_collision_rounds", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(ckpt_args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    loader = DataLoader(data, num_workers=getattr(ckpt_args, "num_workers", 0), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for batch in tqdm(loader, desc="Initial encoding"):
        batch = batch.to(device)
        indices = model.get_indices(batch, use_sk=False).view(-1, model.rq.num_quantizers).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    collision_round = 0
    while collision_round < args.max_collision_rounds and not check_collision(all_indices_str):
        for collision_items in get_collision_item(all_indices_str):
            batch = data[collision_items].to(device)
            indices = model.get_indices(batch, use_sk=True).view(-1, model.rq.num_quantizers).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        collision_round += 1

    print("All indices number:", len(all_indices))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))
    collision_rate = (len(all_indices_str) - len(set(all_indices_str.tolist()))) / len(all_indices_str)
    print("Collision Rate", collision_rate)

    payload = {str(item): list(indices) for item, indices in enumerate(all_indices.tolist())}
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(args.output_file)


if __name__ == "__main__":
    main()

