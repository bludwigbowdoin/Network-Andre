import streamlit as st
import numpy as np
from os import system
import random
import re
import spacy
from spacy.matcher import Matcher
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2model import *
from andre import *


EPISODES = ['season2ep1.txt', 'season2ep3.txt', 'season5ep5.txt', \
    'season5ep6.txt', 'season25ep1356.txt']
VOICES = ["Fred", "Ralph", "Trinoids", "Whisper", "Zarvox"]
SEEDS = ["I want to be your sunset eyes, \n those blue skies, your perfect starry night", \
    "Throw rocks at my window, \n Hold the boom box up high. ", \
        "I know you love cheesy love songs, \n So here’s one for you my dear", \
            "I wish I could write a book, \n It would be about me and you", \
                "Roses are red, violets are blue.",\
                    "I know its a cliche to say how time flies when I'm with you."]

nlp = spacy.load("en_core_web_lg")
matcher = Matcher(nlp.vocab)

# Streamlit page config
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
    min_value=1, max_value=500, value=30)

generations = st.number_input("Number of content generations (max 5000)", \
    min_value=1, max_value=5000, value=50)


temperature = st.slider("Temperature (how much Eric Andre)", \
    min_value=0.0, max_value=1.0, value=0.5, step=0.05)


if st.button('Generate!'):
    
    st.image("./ericFace.gif")

    generated_text = generate_some_text(seed, text_len)
    poetry_doc = nlp(generated_text)
    poetry_useful_words = andre_doc.lemmatize_useful_words(poetry_doc)
    poetry_score = sentence_score(poetry_useful_words)

    # worst and best by sentence score 
    worst_text_swapped = ""
    best_text_swapped = ""
    worst_score = 100000000000
    best_score = 0
    worst_speech = ""
    best_speech = ""
    sentence_scores = []


    # worst and best by vector score
    worst_text_vector = ""
    best_text_vector = ""
    worst_vector_score = 1.0
    best_vector_score = 0.0
    worst_vector_speech = ""
    best_vector_speech = ""
    vector_scores = []

    for i in range(generations):

        swapped_output = andre_doc.swap_within_pos(poetry_doc, temperature)
        swapped_output = swapped_output.lower()
        swapped_output = swapped_output.replace('"', '')
        swapped_output = swapped_output.replace("'", "")
        rand_voice = random.choice(VOICES)
        speech_text = "say -v " + rand_voice + " \"" + swapped_output + "\""
        swapped_doc = nlp(swapped_output)
        swapped_useful_words = andre_doc.lemmatize_useful_words(swapped_doc)

        swapped_score = sentence_score(swapped_useful_words)
        sentence_scores.append(swapped_score)

        vector_sum_score = vector_addition_score(swapped_useful_words, \
            poetry_useful_words)
        vector_scores.append(vector_sum_score)
        
        if swapped_score > best_score: 
            best_score = swapped_score
            best_text_swapped = swapped_output
            best_speech = speech_text
        
        elif swapped_score < worst_score: 
            worst_score = swapped_score
            worst_text_swapped = swapped_output
            worst_speech = speech_text

        if vector_sum_score > best_vector_score:
            best_vector_score = vector_sum_score
            best_text_vector = swapped_output
            best_vector_speech = speech_text

        elif vector_sum_score < worst_vector_score:
            worst_vector_score = vector_sum_score
            worst_text_vector = swapped_output
            worst_vector_speech = speech_text

        if developer:
            st.write("GPT-2 text: \n" + generated_text)
            st.write("Internal sentence score: ", poetry_score)
            st.write(swapped_output)
            st.write("Internal sentence score: ", swapped_score)
            st.write("Vector sum score between poetry and generated text: ", \
                vector_sum_score)
            st.write("-----------------------------------------------")


    st.header("The worst and the best:")
    st.write("Worst by sentence score:")
    st.write(worst_text_swapped)
    st.write("Internal sentence score: ", worst_score)

    st.write("Best by sentence score:")
    st.write(best_text_swapped)
    st.write("Internal sentence score: ", best_score)

    st.write("Worst by vector sum:")
    st.write(worst_text_vector)
    st.write("Vector sum score: ",  worst_vector_score)

    st.write("Best by vector sum:")
    st.write(best_text_vector)
    st.write("Vector sum score: ", best_vector_score)


    if developer:
        fig, score_data = plt.subplots()
        score_data.scatter(sentence_scores, vector_scores)   
        score_data.set_xlabel("Internal Sentence Score")
        score_data.set_ylabel("Vector Sum Score")
        score_data.set_ybound(0,1.1)
        st.pyplot(fig)


    for i in range(3):
        rand_voice = random.choice(VOICES)
        speech_text = "say -v " + rand_voice + " \" loading \""
        system(speech_text)
        rand_voice = random.choice(VOICES)
        speech_text = "say -v " + rand_voice + " \" waiting \""
        system(speech_text)
        rand_voice = random.choice(VOICES)
        speech_text = "say -v " + rand_voice + " \" thinking \""
        system(speech_text)

    system(worst_speech)
    system(best_speech)
    system(worst_vector_speech) 
    system(best_vector_speech)

    st.image("./endOfShow.jpg")
    