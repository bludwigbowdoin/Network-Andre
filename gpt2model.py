"""
Bjorn Ludwig
CSCI 3725
M6: Poetry Slam
11/22/2022

This file contains neural network functions based in PyTorch. All of the code 
in this file comes from the github repo (url below). I only adjusted the work 
from that repo by adjusting the size of the gpt2 model and removing unnecessary 
imports and functions for my purposes. Comments and documentation are original 
to the creator of the repo. 

gist.github.com/mf1024/430d7fd6ff527350d3e4b5bda0d8614e#file-gpt2-medium_text_gen-ipynb
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
model = model.to(device)

def choose_from_top(probs, n=5):
    """
    Function to first select topN tokens from the probability list and then
    based on the selected N word distribution get random token ID.
    """
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate_some_text(input_str, text_len = 250):

    cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)
    model.eval()

    with torch.no_grad():
        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            #Take the first (only one) batch and the last predicted embedding
            softmax_logits = torch.softmax(logits[0,-1], dim=0) 
            #Randomly (from the given probability distribution) 
            #   choose the next word from the top n words
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), \
                n=10) 
            cur_ids = torch.cat([cur_ids, \
                torch.ones((1,1)).long().to(device) * \
                    next_token_id], dim = 1) # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        
        return output_text

