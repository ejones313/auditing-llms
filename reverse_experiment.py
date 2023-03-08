from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from utils import to_jsonl, get_output_file, load_outputs
from args_utils import parse_args
from arca import run_arca
from gbda import run_gbda
from model_utils import get_raw_embedding_table, get_model_and_tokenizer

ATTACKS_DICT = {'arca': run_arca, 'autoprompt': run_arca, 'gbda': run_gbda}

def run_opts(args, model, tokenizer, embedding_table, infile):
    output_targets = load_outputs(infile) 
    if args.max_num_examples is not None:
        output_targets = output_targets[:args.max_num_examples]
    output_filename = get_output_file(infile, output_dir = 'reverse_outputs')
    results_dicts = []
    for i, output_target in tqdm(enumerate(output_targets), total = len(output_targets)):
        results_dict = {}
        results_dict['output_target'] = output_target
        for attack_name in args.opts_to_run:
            assert attack_name in ATTACKS_DICT
            if attack_name == 'arca':
                args.autoprompt = False
            elif attack_name == 'autoprompt':
                args.autoprompt = True
            prompts = []
            n_iters = []
            attack_times = []
            all_prompt_toks = []
            metadata = defaultdict(list)
            successes = 0
            for trial in range(args.n_trials):
                attack_fn = ATTACKS_DICT[attack_name]
                start = datetime.now()
                prompt_toks, n_iter, run_metadata = attack_fn(args, model, tokenizer, embedding_table, output_str = output_target)
                if n_iter == -1:
                    prompt = None
                else:
                    prompt = tokenizer.decode(prompt_toks)
                    prompt_toks = list(prompt_toks)
                    successes += 1
                prompts.append(prompt)
                all_prompt_toks.append(prompt_toks)
                n_iters.append(n_iter)
                attack_times.append((datetime.now() - start).seconds)
                for key in run_metadata:
                    metadata[key].append(run_metadata[key])
            # Log results 
            results_dict[f'{attack_name}'] = {}
            results_dict[f'{attack_name}']['prompts'] = prompts
            results_dict[f'{attack_name}']['prompt_toks'] = all_prompt_toks
            results_dict[f'{attack_name}']['iters'] = n_iters
            results_dict[f'{attack_name}']['time'] = attack_times 
            results_dict[f'{attack_name}']['success_rate'] = successes / args.n_trials
            for key in metadata:
                results_dict[f'{attack_name}'][key] = metadata[key]
        results_dicts.append(results_dict)
        if (i + 1) % args.save_every == 0:
            print("Saving...")
            all_dicts = [vars(args)] + results_dicts
            to_jsonl(all_dicts, output_filename)

    all_dicts = [vars(args)] + results_dicts
    to_jsonl(all_dicts, output_filename)

if __name__ == '__main__':
    args = parse_args()
    model, tokenizer = get_model_and_tokenizer(args)
    embedding_table = get_raw_embedding_table(model)
    infile = f'data/{args.filename}'
    run_opts(args, model, tokenizer, embedding_table, infile)
