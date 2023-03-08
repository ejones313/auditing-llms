import torch
import numpy as np
import pickle
import os
import json
from datetime import datetime

def load_outputs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    for line in lines:
        if not line.startswith(' '):
            print(f"Warning: output {line} doesn't have a preceeding whitespace")
    return lines

def get_str_time():
    time = datetime.now()
    str_time = time.strftime('%Y-%m-%d-%H:%M:%S:%f')
    return str_time

def get_output_file(name, output_dir = 'joint_opt_outputs', file_type = 'jsonl'):
    datetime_str = get_str_time()
    return os.path.join(output_dir, f'{name}_{datetime_str}.{file_type}')

def get_idx(string, l):
    for i, elem in enumerate(l):
        if elem == string:
            return i
    assert False

def restrict_vocab(og_embeddings, toks_to_ignore):
    new_tok_ids = np.array([i for i in range(og_embeddings.shape[0]) if i not in toks_to_ignore])
    embeddings = og_embeddings[new_tok_ids]
    return embeddings, new_tok_ids

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def to_jsonl(dicts, save_file):
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'w') as f:
        for line_dict in dicts:
            print(line_dict)
            jsonl_line = f'{json.dumps(line_dict, cls = NpEncoder)}\n'
            f.write(jsonl_line)

def get_unigram_probs(constraint, device = 'cuda', gptj = False):
    neg_constraint = constraint.startswith('not')
    if neg_constraint:
        constraint = constraint[len('not_'):]
    # Constraints taken from: https://github.com/unitaryai/detoxifysssssss
    tox_constraints = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    languages = ['en', 'es', 'fr', 'it', 'de']
    if constraint in tox_constraints:
        with open('extra_files/tox_log_probs.pkl', 'rb') as f:
            log_probs = pickle.load(f)
        idx = get_idx(constraint, tox_constraints)
        unigram_probs = log_probs[:, idx]
    elif constraint in languages:
        with open(f'extra_files/{constraint}_logprobs.pkl', 'rb') as f:
            unigram_probs = pickle.load(f)
    else:
        raise NotImplementedError
    if neg_constraint:
        unigram_probs = np.log(1 - np.exp(unigram_probs))
    if gptj:
        # Rule out the extra tokens
        unigram_probs = np.concatenate([unigram_probs, -10000 * np.ones(50400 - 50257)], axis = 0)
        print("Unigrams prob shape: ", unigram_probs.shape)
    return torch.Tensor(unigram_probs).to(device)


def get_forbidden_toks(args, tokenizer, n_total_toks = 50257, output = False, output_str = None):
    constraint = args.inpt_tok_constraint if not output else args.output_tok_constraint
    if constraint is None:
        if not output and output_str is not None:
            return toks_to_skip(tokenizer(output_str)['input_ids'], tokenizer, n_total_toks)
        else:
            return set()
    top_k = args.top_k_input if not output else args.top_k_output
    #constraints of the form not_toxic
    neg_constraint = constraint.startswith('not')
    if neg_constraint:
        constraint = constraint[len('not_'):]
    if constraint.startswith('toxic'):
        with open(f'extra_files/most_{constraint}.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('spanish'):
        assert not neg_constraint
        with open(f'extra_files/es.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('english'):
        assert not neg_constraint
        with open(f'extra_files/en.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('german'):
        assert not neg_constraint
        with open(f'extra_files/de.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('french'):
        assert not neg_constraint
        with open(f'extra_files/fr.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('italian'):
        assert not neg_constraint
        with open(f'extra_files/it.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('longest'):
        assert top_k is not None
        with open(f'extra_files/longest.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    elif constraint.startswith('lowercase'):
        toks = [tokenizer.decode([i]) for i in range(n_total_toks)]
        constraint_toks = []
        for i in range(n_total_toks):
            if toks[i] == toks[i].lower():
                constraint_toks.append(i)
        constraint_toks = np.array(constraint_toks)
    elif constraint == 'letters':
        with open(f'extra_files/letter_toks.pkl', 'rb') as f:
            constraint_toks = pickle.load(f)
    else:
        raise NotImplementedError
    if top_k != 0:
        constraint_toks = constraint_toks[:top_k]
    if not neg_constraint:
        constraint_toks = filter_forbidden_toks(np.arange(n_total_toks), constraint_toks)
    if not output and output_str is not None:
        deg_constraint_toks = toks_to_skip(tokenizer(output_str)['input_ids'], tokenizer, n_total_toks)
        # Fine to have duplicates, since this gets passed into filter_forbidden_toks
        constraint_toks = np.concatenate([constraint_toks, deg_constraint_toks], axis = 0)
        print("Adding output toks!")
        assert False
    return constraint_toks

def filter_forbidden_toks(toks_tensor, forbidden_toks):
    if len(forbidden_toks) == 0:
        return toks_tensor
    # Toks tensor has all tokens included
    mask = np.zeros(toks_tensor.shape[0])
    # Should try to get the indices where bad things happen...
    mask[forbidden_toks] = 1
    if isinstance(toks_tensor, torch.Tensor):
        elements_ok = np.where(mask[toks_tensor.detach().cpu().numpy()] == 0)[0]
    else:
        elements_ok = np.where(mask[toks_tensor] == 0)[0]
    toks_tensor = toks_tensor[elements_ok]
    return toks_tensor

def toks_to_skip(output_toks, tokenizer, n_total_toks = 50257):
    toks_to_skip = []
    if isinstance(output_toks, torch.Tensor):
        output_toks = output_toks.detach().cpu().numpy()
    all_toks = [tokenizer.decode([i]) for i in range(n_total_toks)]
    output_tok_strs = [all_toks[i] for i in output_toks]
    for i, tok in enumerate(all_toks):
        if len(tok) <= 3 and tok not in output_tok_strs:
            continue
        # token is fair-game to elimate
        for otok in output_tok_strs:
            otok = otok.strip(' ').lower()
            tok = tok.strip(' ').lower()
            # Asymmetric case: remove one letter off of the target tok, but not the output tok...
            if tok.startswith(otok[:-1]) or otok.startswith(tok):
                toks_to_skip.append(i)
    return np.array(toks_to_skip)
