import streamlit as st
import numpy as np
from os import system
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


EPISODES = ['season2ep1.txt', 'season2ep3.txt', 'season5ep5.txt', \
    'season5ep6.txt', 'season25ep1356.txt']
VOICES = ["Bells", "Bad News", "Fred", "Ralph", "Trinoids", "Whisper", "Zarvox"]
SEEDS = ["I want to be your sunset eyes, \n those blue skies, your perfect starry night", \
    "Throw rocks at my window, \n Hold the boom box up high. ", \
        "I know you love cheesy love songs, \n So here’s one for you my dear", \
            "I wish I could write a book, \n It would be about me and you", \
                "Roses are red, violets are blue.",\
                    "I know its a cliche to say how time flies when I'm with you."]

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Streamlit Page Config
st.set_page_config(
     page_title="Network Andre",
     page_icon="📹",
 )
st.title("Network Andre")
st.image("./ericAndreShow.jpg")

developer = st.checkbox('Developer mode')   # dev mode toggle



episode_title = st.selectbox('Source episode', EPISODES)



andre_doc = Andre(nlp, episode_title)
andre_doc.set_text()
sent_list = list(andre_doc.sentences())
rand_sent = random.choice(sent_list)



if st.checkbox("Write your own seed"): 
    seed = st.text_input("Seed")

else:   
    seed = st.selectbox("Seed", SEEDS)

text_len = st.number_input("Approximate length of response (in words)", \
    min_value=1, max_value=500, value=25)

temperature = st.slider("Temperature (how much Eric Andre)", \
    min_value=0.0, max_value=1.0, value=0.5, step=0.05)


if st.button('Generate!'):
    
    st.image("./ericFace.gif")

    generated_text = generate_some_text(seed, text_len)
    poetry_doc = nlp(generated_text)
    poetry_score = sentence_score(andre_doc.lemmatize_useful_words(poetry_doc))

    worst_text_swapped = ""
    best_text_swapped = ""
    worst_score = 100000000000
    best_score = 0
    best_speech = ""
    worst_speech = ""

    for i in range(50):

        swapped_output = andre_doc.swap_within_pos(poetry_doc, temperature)
        rand_voice = random.choice(VOICES)
        speech_text = "say -v " + rand_voice + " \"" + swapped_output + "\""
        swapped_doc = nlp(swapped_output)
        swapped_score = sentence_score(andre_doc.lemmatize_useful_words(swapped_doc))
        
        if swapped_score > best_score: 
            best_score = swapped_score
            best_text_swapped = swapped_output
            best_speech = speech_text
        
        elif swapped_score < worst_score: 
            worst_score = swapped_score
            worst_text_swapped = swapped_output
            worst_speech = speech_text
        
        if developer:
            st.write("GPT-2 text: \n" + generated_text)
            st.write("Score: \n", poetry_score)
            st.write(swapped_output)
            st.write(swapped_score)




    st.write("WORST: \n" + worst_text_swapped)
    st.write("score: ", worst_score)
    system(worst_speech)

    st.write("BEST: \n" + best_text_swapped)
    st.write("score: ", best_score)
    system(best_speech)

    st.image("./endOfShow.jpg")

    # for i in range(option_2):
    #     conversation_seed = generate_some_text(conversation_seed, text_len = seqlen)
    #     conversation_seed = generate_some_text(conversation_seed, text_len = seqlen)

    #     # conversation_seed = generate_text(option_3, conversation_seed, developer)
    #     # conversation_seed = generate_text(option_4, conversation_seed, developer)
















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