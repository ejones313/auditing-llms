# Code adapted from https://github.com/facebookresearch/text-adversarial-attack
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from losses import log_prob_loss 
from utils import get_forbidden_toks, restrict_vocab
from utils import toks_to_skip as tts

def run_gbda(args, model, tokenizer, embedding_table, output_str = None):
    if output_str is None:
        raise NotImplementedError
    # initial setup
    run_metadata = {}
    args.batch_size = args.gbda_initializations
    output_toks = torch.Tensor(tokenizer(output_str)['input_ids']).long().to(args.device)
    output_len = output_toks.shape[0]
    run_metadata['n_output_toks'] = output_len
    labels = torch.cat([-100 * torch.ones(args.prompt_length).to('cuda').unsqueeze(0), output_toks.unsqueeze(0)], dim = 1).long()
    labels = labels.repeat(args.batch_size, 1, 1)
    output_embeddings = embedding_table[output_toks].unsqueeze(0)
    output_embeddings = output_embeddings.repeat(args.batch_size, 1, 1)
    if args.keep_close_toks:
        toks_to_skip = output_toks.detach().cpu().numpy()
    else:
        toks_to_skip = tts(output_toks, tokenizer, n_total_toks = 50257)
        if args.inpt_tok_constraint is not None:
            more_forbidden_toks = get_forbidden_toks(args, tokenizer)
            toks_to_skip = np.concatenate([toks_to_skip, more_forbidden_toks], axis = 0)

    embedding_table, new_to_old = restrict_vocab(embedding_table, toks_to_skip)
    new_to_old_tensor = torch.Tensor(new_to_old).long().to('cuda')
    vocab_size = embedding_table.shape[0]
    model.eval()
    with torch.no_grad():
        if args.model_id == 'gptj':
            log_coeffs = torch.randn(args.batch_size, args.prompt_length, vocab_size, dtype = torch.half)
        else:
            log_coeffs = torch.randn(args.batch_size, args.prompt_length, vocab_size)
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad = True
    if args.model_id == 'gptj':
        optimizer = torch.optim.Adam([log_coeffs], lr=args.gbda_learning_rate, eps = 1e-7)
    else:
        optimizer = torch.optim.Adam([log_coeffs], lr=args.gbda_learning_rate, eps = 1e-8)
    for it in tqdm(range(args.gbda_iters)):
        optimizer.zero_grad()
        coeffs = F.gumbel_softmax(log_coeffs, hard = False, dim = 2)
        weighted_embeddings = (coeffs @ embedding_table[None, :, :]) # B x T x D
        full_embeddings = torch.cat([weighted_embeddings, output_embeddings], dim = 1)
        out = model(inputs_embeds = full_embeddings, labels = labels)
        loss = log_prob_loss(out, labels)
        loss.backward(retain_graph = True)
        optimizer.step()
        prompt_toks = log_coeffs.argmax(dim = 2)
        rel_embeddings = torch.stack([embedding_table[prompt_toks[i]] for i in range(args.batch_size)])
        full_embeddings = torch.cat([rel_embeddings, output_embeddings], dim = 1)
        model.eval()
        out = model(inputs_embeds = full_embeddings)
        generated_output_ids = out.logits[:, -1 - output_len: -1, :].argmax(dim = 2)
        for j in range(args.batch_size):
            gen_output = generated_output_ids[j]
            if (output_toks == gen_output).all().item():
                # Success case
                return new_to_old[prompt_toks[j].detach().cpu().numpy()], it, run_metadata
    return -1, -1, run_metadata
