This repository contains code for the following paper:
> [Automatically Auditing Large Language Models via Discrete Optimization](https://arxiv.org/abs/2303.04381)
>
> Erik Jones, Anca Dragan, Aditi Raghunathan, and Jacob Steinhardt 

### Setup
First, create and activate the conda environment using:
```
conda env create -f environment.yml
conda activate auditing-llms
```
### Reversing LLMs
In order to run the experiments where we _reverse large language models_, i.e. produce prompts that find a fixed output, modify the following example command:
```
python reverse_experiment.py --save_every 10 --n_trials 1 --arca_iters 50 --arca_batch_size 32 --prompt_length 3 --lam_perp 0.2 --label your-file-label --filename senators.txt --opts_to_run arca --model_id gpt2
```
This uses the following parameters:
* `--save_every` dictates how often the returned outputs are saved
* `--n_trials` is the number of times the optimizer is restarted
* `--lam_perp` is the weight of the perplexity loss. Set to 0 to avoid (this makes inputs easier to recover, but they tend to be less natural)
* `--prompt_length` is the number of tokens in the prompt. 
* `--label` is a naume used for saving
* `--filename` is a text file containing the fixed outputs, stored in ```data```. We include `senators.txt`, `tox_1tok.txt`, `tox_2tok.txt`, and `tox_3tok.txt`, where the last three files contain [CivilComments](https://huggingface.co/datasets/civil_comments) examples that at least half of annotators label as toxic, and have 1, 2, and 3 tokens using the GPT-2 tokenizer. 
* `--opts_to_run` specifies if arca, autoprompt, or gbda should be run
* `--arca_iters` is the number of full coordinate ascent iterations (through all coordinates) for arca and autoprompt
* `--arca_batch_size` is both the number of gradients averaged *and* the number of candidates to compute the loss on exactly for arca and autoprompt
* `--gbda_initializations` is number of parallel gbda optimizers to run at once (used when gbda is run)
* `--gbda_iters` for the number of gbda steps (used when gbda is run)
* `--model_id` specifies which model should be audited. 
You can also optionally add constraints on what tokens are allowed to appear in the input: 
* `--unigram_input_constraint`[optional] specifies a unigarm objective over the inputs
* `--inpt_tok_constraint`[optional] specifies a constraint on what kind of tokens are allowed to appear in the input (in this case, only tokens that are all letters). 
* `--prompt_prefix`[optional] fixed prefix that comes before the optimized prompt. 

### Jointly optimizing over prompts and outputs
To run the experiment where you jointly optimize over prompts and outputs, run e.g.: 
```
python joint_optimization_experiment.py --save_every 10 --n_trials 100 --arca_iters 50 --arca_batch_size 32 --lam_perp 0.5 --label your-file-label --model gpt2 --unigram_weight 0.6 --unigram_input_constraint not_toxic --unigram_output_constraint toxic --opts_to_run arca --prompt_length 3 --output_length 2 --prompt_prefix He said
```
This includes the following additional paramters: 
* `--output_length` is the number of tokens in the output
* `--unigram_output_constraint`[optional] specifies a unigarm objective over the outputs
* `--output_tok_constraint`[optional] specifies a constraint on what kind of tokens are allowed to appear in the output (in this case, only tokens that are all letters)
