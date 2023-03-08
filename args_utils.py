import argparse

def parse_args(joint = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arca_batch_size', type = int, default = 64, 
            help = 'Number of gradients to average + probabilities to compute exactly')
    parser.add_argument('--n_trials', type = int, default = 1,
            help = 'Number of times to run the attack')
    parser.add_argument('--arca_iters', type = int, default = 10,
            help = 'Number of times to iterate through each coordinate for the linear attack')
    parser.add_argument('--prompt_length', type = int, default = 2,
            help = 'Size of the input prompt')
    parser.add_argument('--save_every', type = int, default = 5,
            help = 'Number of save steps')
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--lam_perp', type = float, default = 0,
            help = "Weight on the perplexity term in the loss")
    parser.add_argument('--label', type = str, default = 'none',
            help = 'Label of the kind of experiment being run, used to collate results (stored in log files)')
    parser.add_argument('--model_id', default = 'gpt2-large', type = str,
            help = "Model to evaluate")
    parser.add_argument('--prompt_prefix', type = str, default = None, nargs='+',
            help = 'Prefix to include before the prompt for optimization')
    parser.add_argument('--inpt_tok_constraint', type = str, default = None,
            help = 'Constraint on the set of allowable input tokens')
    parser.add_argument('--output_tok_constraint', type = str, default = None,
            help = 'Constraint on the set of allowable input output')
    parser.add_argument('--top_k_input', type = int, default = 0,
            help = 'Additionally filter input constraint by best k; e.g. for longest, this gives you the k longest')
    parser.add_argument('--top_k_output', type = int, default = 0,
            help = 'Additionally filter output constraint by best k; e.g. for longest, this gives you the k longest')
    parser.add_argument('--autoprompt', action = 'store_true',
            help = 'Compute gradients at current token')
    parser.add_argument('--unigram_input_constraint', type = str, default = None,
            help = 'Unigram loss on input')
    parser.add_argument('--unigram_output_constraint', type = str, default = None,
            help = 'Unigram loss on output')
    parser.add_argument('--unigram_weight', type = float, default = 1,
            help = 'Weight on the unigram loss...')
    parser.add_argument('--opts_to_run', type = str, nargs='+', 
            default = ['arca', 'gbda'], help = 'Types of attacks to run')
    parser.add_argument('--gbda_initializations', type = int, default = 8, 
            help = 'Number of paralel gbda attacks to run')
    parser.add_argument('--gbda_iters', type = int, default = 100,
            help = 'Number of gradient steps to take for the gbda attack')
    parser.add_argument('--gbda_learning_rate', type = float, default = 1e-1,
            help = 'Learning rate to use for the gbda attack')
    parser.add_argument('--filename', type = str, default = 'senators.txt',
            help = 'File where the outputs to be generated are stored')
    parser.add_argument('--max_num_examples', type = int, default = None,
            help = 'Maximum number of examples to run through')
    parser.add_argument('--output_length', type = int, default = 2,
            help = 'Size of the oiutput target')
    parser.add_argument('--pair_type', type = str, default = None,
            help = 'Either prompt_longer, output_longer, or same_length')
    args = parser.parse_args()
    if isinstance(args.prompt_prefix, list):
        args.prompt_prefix = ' '.join(args.prompt_prefix)
    return args
