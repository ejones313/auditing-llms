from collections import defaultdict
from datetime import datetime

from args_utils import parse_args
from arca import run_arca 
from model_utils import get_raw_embedding_table, get_model_and_tokenizer
from utils import to_jsonl, get_output_file


def run_opts(args, model, tokenizer, embedding_table, hparam_dicts):
    results_dicts = []
    # First line of the output stores the arguments, rest store different output files
    output_filename = get_output_file(args.label, output_dir = 'joint_opt_outputs')
    for attack_name in args.opts_to_run:
        assert attack_name in ['autoprompt', 'arca']
        if attack_name == 'arca':
            args.autoprompt = False
        else:
            args.autoprompt = True
        for i, hparam_dict in enumerate(hparam_dicts):
            for key in hparam_dict:
                setattr(args, key, hparam_dict[key])
            results_dict = {}
            results_dict['hparams'] = hparam_dict 
            prompt_output_pairs = []
            n_iters = []
            opt_times = []
            all_prompt_output_toks = []
            metadata = defaultdict(list)
            successes = 0
            for trial in range(args.n_trials):
                start = datetime.now()
                ret_toks, n_iter, run_metadata = run_arca(args, model, tokenizer, embedding_table)
                if n_iter == -1:
                    prompt = None
                    output = None
                else:
                    prompt = tokenizer.decode(ret_toks[:-args.output_length])
                    output = tokenizer.decode(ret_toks[-args.output_length:])
                    ret_toks = list(ret_toks)
                    successes += 1
                prompt_output_pairs.append((prompt, output))
                all_prompt_output_toks.append(ret_toks)
                n_iters.append(n_iter)
                opt_times.append((datetime.now() - start).seconds)
                # Log results 
                for key in run_metadata:
                    metadata[key].append(run_metadata[key])
                results_dict[f'{attack_name}'] = {}
                results_dict[f'{attack_name}']['prompt_output_pairs'] = prompt_output_pairs
                results_dict[f'{attack_name}']['toks'] = all_prompt_output_toks
                results_dict[f'{attack_name}']['iters'] = n_iters
                results_dict[f'{attack_name}']['time'] = opt_times 
                results_dict[f'{attack_name}']['success_rate'] = successes / (trial + 1)
                for key in metadata:
                    results_dict[f'{attack_name}'][key] = metadata[key]
                if (trial + 1) % args.save_every == 0:
                    all_dicts = [vars(args)] + results_dicts + [results_dict]
                    to_jsonl(all_dicts, output_filename)
            results_dicts.append(results_dict)
            all_dicts = [vars(args)] + results_dicts
            to_jsonl(all_dicts, output_filename)
    all_dicts = [vars(args)] + results_dicts
    to_jsonl(all_dicts, output_filename)

if __name__ == '__main__':
    args = parse_args(joint = True)
    model, tokenizer = get_model_and_tokenizer(args)
    embedding_table = get_raw_embedding_table(model)
    hparam_dicts = []
    pairs = []
    if args.pair_type is not None:
        if args.pair_type == 'same_length':
            pairs = [(2,2),(3,3),(4,4),(5,5),(6,6)]
        elif args.pair_type == 'output_longer':
            pairs = [(2,3),(3,4),(4,5),(5,6),(6,7)]
        elif args.pair_type == 'prompt_longer':
            pairs = [(2,1),(3,2),(4,3),(5,4),(6,5)]
        else:
            raise NotImplementedError
    else:
        pairs = [(args.prompt_length, args.output_length)]
    for (pl, ol) in pairs:
        hparam_dict = {}
        hparam_dict['prompt_length'] = pl
        hparam_dict['output_length'] = ol
        hparam_dicts.append(hparam_dict)
    print(f"Running {len(hparam_dicts)} sets of hyperparameters")
    run_opts(args, model, tokenizer, embedding_table, hparam_dicts)
