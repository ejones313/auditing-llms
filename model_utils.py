from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTJForCausalLM, AutoTokenizer
from datetime import datetime
import torch

def get_raw_embedding_table(model):
    return model.get_input_embeddings()._parameters['weight']

def get_model_and_tokenizer(args):
    print("Loading model and tokenizer...")
    start = datetime.now()
    if args.model_id.startswith('gpt2'):
        model = GPT2LMHeadModel.from_pretrained(args.model_id)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_id)
    elif args.model_id == 'gptj':
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        raise NotImplementedError
    model = model.to('cuda')
    print(f"Finished in {str(datetime.now()-start)}")
    return model, tokenizer
