import re
import json
import numpy as np
import torch

def clean_expression(expression):
    expression_cleaned = re.sub(r"([.,'!?\"()*#:;])", '', expression.lower())
    expression_cleaned = expression_cleaned.replace('-', ' ')
    expression_cleaned = expression_cleaned.replace('/',' ')
    return expression_cleaned.split()

def build_vocab(ann_file):
    anns_all = json.load(open(ann_file, 'r'))
    token2idx = {
        "PAD": 0,
        "UNK": 1,
    }

    for split_name in anns_all:
        for ann in anns_all[split_name]:
            for expression in ann['expressions']:
                words = clean_expression(expression)
                for word in words:
                    if word not in token2idx:
                        token2idx[word] = len(token2idx)

    idx2token = {idx: token for token, idx in token2idx.items()}

    return token2idx, idx2token

def build_glove_matrix(token2idx, glove_model, glove_dim):
    vocab_size = len(token2idx)
    weight_matrix = np.random.uniform(-0.01,0.01, (vocab_size, glove_dim)).astype(np.float32)
    weight_matrix[0] = np.zeros(glove_dim)

    found =0
    for word, idx in token2idx.items():
        if word in glove_model:
            weight_matrix[idx] = glove_model[word]
            found += 1
    print(f"Tìm được {found} trên {vocab_size} từ trong glove")

    return torch.from_numpy(weight_matrix)

def tokenize_expression(expression, token2idx, max_token) :
    ref_inds = torch.zeros(max_token, dtype=torch.long)
    words = clean_expression(expression)

    for i, word in enumerate(words):
        if i >= max_token:
            break
        if word in token2idx:
            ref_inds[i] = token2idx[word]
        else:
            ref_inds[i] = token2idx['UNK']

    return ref_inds