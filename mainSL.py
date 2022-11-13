import streamlit as st
import numpy as np
import os
import random
import re
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2model import *

# Streamlit Page Config
st.set_page_config(
     page_title="Network Andre",
     page_icon="📹",
 )

st.title("Network Andre")
st.image("./ericAndreShow.jpg", caption="The Eric Andre Show")
developer = st.checkbox('Developer Mode')   # dev mode toggle

# Data prep
episode_dict = {'season25ep1356.txt':'', 'season2ep1.txt':'', 'season2ep3.txt':'', 'season5ep5.txt':'', 'season5ep6.txt':''}

for episode in episode_dict.keys():
    with open(episode) as f:
        contents = re.sub(r'someword=|\,.*|\#.*','', f.read())
        contents = re.sub(r'\n+', '\n', contents).strip()
        # contents = f.read()
        episode_dict[episode] = contents
        
episodes = list(episode_dict.keys())

# gpt2 setup
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

print(generate_some_text("Hello good sir, how do you ", text_len = 10))



# Just some fun
loading_messages = ["Waiting..."]

# Constants
if developer:
    seqlen = st.number_input('seqlen', value=100)
    lstm_diversity = st.number_input('lstm_diversity', value=0.2)
    lstm_max_length = st.number_input('lstm_max_length', value=250)
    gpt3_temperature = st.number_input('gpt3_temperature', value=0.1)
else:
    seqlen = 100
    lstm_diversity = 0.2
    lstm_max_length = 250
    gpt3_temperature = 0.1

# Generate Text Function
def generate_text(episode, seed, debug):

    text = episode_dict[episode]

    # model_loc = "./modelsAndre/" + philosopher_name + "v5"
    model_loc = "./modelsAndre/" + "season25ep1356.txtv1"
    model = keras.models.load_model(model_loc)

    if debug:
        st.header(episode)
        st.write("Prompt: ", seed)

    while len(seed) < seqlen:
        pretext = ""
        for i in range(seqlen-len(seed)):
            pretext = pretext + " "
        seed = pretext + seed

    seed = seed[len(seed)-seqlen:]
    print("Seed is: ", seed)

    if debug:
        st.write("Seed: ", seed)

    diversity = lstm_diversity
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.exp(np.log(preds) / temperature)  # softmax
        preds = preds / np.sum(preds)                #
        probas = np.random.multinomial(1, preds, 1)  # sample index
        return np.argmax(probas)    

    response_text = ""
    next_char = ""
    i = 0

    message = random.choice(loading_messages)
    with st.spinner(message):
        while (i < lstm_max_length):
                x_pred = np.zeros((1, seqlen, len(chars)))
                for t, char in enumerate(seed):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)
                next_index = sample(preds[0, -1], diversity)
                next_char = indices_char[next_index]

                seed = seed[1:] + next_char

                response_text = response_text + next_char
                i = i + 1

        if debug:
            st.write("LSTM: ", response_text)


    return response_text


option_2 = st.slider('How many dialogs should we generate?', 1, 5, 1)
option_3 = st.selectbox('Select Episode 1', episodes)
option_4 = st.selectbox('Select Episode 2', episodes)

conversation_seed = st.text_input("Prompt", "The soul is")

if st.button('Generate!'):

    for i in range(option_2):
        conversation_seed = generate_text(option_3, conversation_seed, developer)
        conversation_seed = generate_text(option_4, conversation_seed, developer)

    st.balloons()
    st.header("Tusinde Tak")