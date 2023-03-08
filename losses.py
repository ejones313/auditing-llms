import torch
from torch import nn
import torch.nn.functional as F

def log_prob_loss(output, labels, temp = 1, ret_all = False, just_logit = False):
    loss_fct = nn.CrossEntropyLoss(reduction = 'mean')
    logits = output.logits
    if torch.isnan(logits).any():
        assert False
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits / temp
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def log_perplexity(output, prompts, prefix_len = None, ret_all = False):
    shift_prompts = prompts[:, 1:]
    shift_logits = output.logits[:, :shift_prompts.shape[1], :]
    log_probs = F.log_softmax(shift_logits, dim = 2)
    stacked_perplexities = torch.stack([log_probs[i, torch.arange(shift_prompts.shape[1]), shift_prompts[i]].mean() for i in range(log_probs.shape[0])])
    if ret_all:
        return -stacked_perplexities
    return -stacked_perplexities.mean()
