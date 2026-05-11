import json
import csv
import ast
import random
from collections import defaultdict

paths = {
    'Baseline (Original Semantic)': './data/Amazon/index/Industrial_and_Scientific.index.json',
    'R690b (L2 Contrastive LMH)': '/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001.index.json',
    'V2 Tokenizer': '/data/leejt/OneRec/output_weights/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json'
}

# 1. Load Transitions from training data
train_file = 'data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv'
transitions = []

with open(train_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            history = ast.literal_eval(row['history_item_id'])
            target = int(row['item_id'])
            if len(history) > 0:
                last_item = int(history[-1])
                transitions.append((last_item, target))
        except:
            continue

print(f"Total True Transitions (A->B) from train set: {len(transitions)}")

random.seed(42)
all_items = list(set([str(a) for a,b in transitions] + [str(b) for a,b in transitions]))
random_transitions = [(random.choice(all_items), random.choice(all_items)) for _ in range(len(transitions))]

def evaluate_purity(index_path, name):
    with open(index_path, 'r') as f:
        data = json.load(f)
    
    # standardize to list of strings
    codebook = {}
    for k, v in data.items():
        if isinstance(v, str):
            codebook[str(k)] = v.split()
        else:
            codebook[str(k)] = [str(x) for x in v]
            
    def compute_overlap(pairs):
        l1_match = 0
        l2_match = 0
        l3_match = 0
        valid = 0
        for a, b in pairs:
            a_str, b_str = str(a), str(b)
            if a_str in codebook and b_str in codebook:
                valid += 1
                t_a = codebook[a_str]
                t_b = codebook[b_str]
                
                match_depth = 0
                for idx in range(min(len(t_a), len(t_b))):
                    if t_a[idx] == t_b[idx]:
                        match_depth += 1
                    else:
                        break
                
                if match_depth >= 1: l1_match += 1
                if match_depth >= 2: l2_match += 1
                if match_depth >= 3: l3_match += 1
                
        if valid == 0: return 0,0,0
        return (l1_match/valid)*100, (l2_match/valid)*100, (l3_match/valid)*100

    t_l1, t_l2, t_l3 = compute_overlap(transitions)
    r_l1, r_l2, r_l3 = compute_overlap(random_transitions)
    
    print(f"=== {name} ===")
    print(f"Random Pairs Match -> L1: {r_l1:.2f}% | L1+L2: {r_l2:.2f}% | L1+L2+L3: {r_l3:.2f}%")
    print(f"True CF Trans Match-> L1: {t_l1:.2f}% | L1+L2: {t_l2:.2f}% | L1+L2+L3: {t_l3:.2f}%")
    print(f"Lift (Purity Gain) -> L1: {t_l1 - r_l1:+.2f}% | L1+L2: {t_l2 - r_l2:+.2f}%")
    print("-" * 50)

for name, path in paths.items():
    evaluate_purity(path, name)
