import streamlit as st
import numpy as np
import os
import random
import re
import spacy
from spacy.matcher import Matcher
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2model import *
from andre import *


EPISODES = ['season25ep1356.txt', 'season2ep1.txt', \
            'season2ep3.txt', 'season5ep5.txt', 'season5ep6.txt']
VOICES = ["Bells", "Bad News", "Fred", "Ralph", "Trinoids", "Whisper", "Zarvox"]

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Streamlit Page Config
st.set_page_config(
     page_title="Network Andre",
     page_icon="📹",
 )
st.title("Network Andre")
st.image("./ericAndreShow.jpg", caption='The set of "The Eric Andre Show"')

episode_title = st.selectbox('Select Source Episode', EPISODES)



andre = Andre(nlp, episode_title)
andre.set_text()
sent_list = list(andre.sentences())
rand_sent = random.choice(sent_list)


# for sent in sent_list:
#     st.write(sent)
#     rand_voice = random.choice(VOICES)
#     st.write("say -v " + rand_voice + " \"" + str(sent)+ "\"")
#     os.system("say -v " + rand_voice + " \"" + str(sent)+ "\"")


seed = st.text_input("Prompt", "Roses are red, violets are blue, ")
text_len = st.number_input("Length of response (in characters)", \
    min_value=1, max_value=500, value=50)


if st.button('Generate!'):
    generated_text = generate_some_text(seed, text_len)
    rand_voice = random.choice(VOICES)
    speech_text = "say -v " + rand_voice + " \"" + generated_text + "\""
    st.write(speech_text)
    # st.system(speech_text)


   
    # for i in range(option_2):
    #     conversation_seed = generate_some_text(conversation_seed, text_len = seqlen)
    #     conversation_seed = generate_some_text(conversation_seed, text_len = seqlen)

    #     # conversation_seed = generate_text(option_3, conversation_seed, developer)
    #     # conversation_seed = generate_text(option_4, conversation_seed, developer)

    st.balloons()
















# # Generate Text Function
# def generate_text(episode, seed, debug):

#     text = episode_dict[episode]

#     # model_loc = "./modelsAndre/" + philosopher_name + "v5"
#     model_loc = "./modelsAndre/" + "season25ep1356.txtv1"
#     model = keras.models.load_model(model_loc)

#     if debug:
#         st.header(episode)
#         st.write("Prompt: ", seed)

#     while len(seed) < seqlen:
#         pretext = ""
#         for i in range(seqlen-len(seed)):
#             pretext = pretext + " "
#         seed = pretext + seed

#     seed = seed[len(seed)-seqlen:]
#     print("Seed is: ", seed)

#     if debug:
#         st.write("Seed: ", seed)

#     diversity = lstm_diversity
#     chars = sorted(list(set(text)))
#     char_indices = dict((c, i) for i, c in enumerate(chars))
#     indices_char = dict((i, c) for i, c in enumerate(chars))

#     def sample(preds, temperature=1.0):
#         preds = np.asarray(preds).astype('float64')
#         preds = np.exp(np.log(preds) / temperature)  # softmax
#         preds = preds / np.sum(preds)                #
#         probas = np.random.multinomial(1, preds, 1)  # sample index
#         return np.argmax(probas)    

#     response_text = ""
#     next_char = ""
#     i = 0

#     message = random.choice(loading_messages)
#     with st.spinner(message):
#         while (i < lstm_max_length):
#                 x_pred = np.zeros((1, seqlen, len(chars)))
#                 for t, char in enumerate(seed):
#                     x_pred[0, t, char_indices[char]] = 1.

#                 preds = model.predict(x_pred, verbose=0)
#                 next_index = sample(preds[0, -1], diversity)
#                 next_char = indices_char[next_index]

#                 seed = seed[1:] + next_char

#                 response_text = response_text + next_char
#                 i = i + 1

#         if debug:
#             st.write("LSTM: ", response_text)


#     return response_text