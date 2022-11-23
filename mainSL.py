"""
Bjorn Ludwig
CSCI 3725
M6: Poetry Slam
11/22/2022

This file operates as both the "main" executer of the functions from other 
files as well as the Streamlit builder. Many of the lines in this file are 
rather trivial st.write statements, and I moved as many as I could into helper 
functions, but some would not behave unless written explicitly in the code. 
There are many lines to this file, but the majority of the algorithm based 
thinking is done in the other 3 .py files. 

To execute this file, type 'streamlit run mainSL.py' into the command line 
while in the Poetry-Slam directory.            
"""

import streamlit as st
import numpy as np
from os import system
import random
import re
import spacy
import matplotlib.pyplot as plt
import time
from gpt2model import *
from andre import *

EPISODES = ['season2ep1.txt', 'season2ep3.txt', 'season5ep5.txt', \
    'season5ep6.txt', 'season25ep1356.txt']
VOICES = ["Fred", "Ralph", "Trinoids", "Zarvox"]
SEEDS = ["What is the meaning of life, Eric?", \
    "Now listen to the following riddle: ", \
        "I want to be your sunset eyes, \n those blue skies, your \
perfect starry night", \
            "Throw rocks at my window, \n Hold the boom box up high. ", \
                "I know you love cheesy love songs, \n So here's one for \
you my dear", \
                    "I wish I could write a book, \n It would be about \
me and you", \
                        "Roses are red, violets are blue.",\
                            "I know its a cliche to say how time flies \
when I'm with you."]


def developer_text(generated_text, poetry_score, swapped_output, \
    swapped_score, vector_sum_score):
    """
    Simple helper function to write output when developer mode is on. 
    """
    st.write("GPT-2 text: \n" + generated_text)
    st.write("Internal sentence relevance score: ", poetry_score)
    st.write(swapped_output)
    st.write("Internal sentence relevance score: ", swapped_score)
    st.write("Vector sum (plagiarism) score between poetry and generated text: ", \
        vector_sum_score)
    st.write("-----------------------------------------------")


def save_poetry(worst_text_swapped, worst_score, best_text_swapped, \
    best_score, worst_text_vector, worst_vector_score, best_text_vector, \
        best_vector_score, generated_text,text_len, generations, \
            temperature, episode_title):
    """
    Simple helper function to save the output when the user checks box to 
    save output as a .txt file. 
    """
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = 'poetry_from_' + curr_time + episode_title
    with open("output/" + file_name, 'w') as f:
        f.write("text-len: " + str(text_len) + ", generations: " + \
            str(generations) + ", temperature: " + str(temperature))
        f.write("\n \n GPT-2 text: \n" + generated_text)
        f.write("\n ----------------------------------------------------- \n")
        f.write("\n \n Worst by sentence relevance score: \n")
        f.write(str(worst_text_swapped) + "\n")
        f.write("Internal sentence relevance score: " + str(worst_score))
        f.write("\n ----------------------------------------------------- \n")
        f.write("\n \n Best by sentence relevance score: \n")
        f.write(str(best_text_swapped)+ "\n")
        f.write("\n Internal sentence relevance score: " + str(best_score))
        f.write("\n ----------------------------------------------------- \n")
        f.write("\n \n Worst by vector sum (plagiarism): \n")
        f.write(str(worst_text_vector)+ "\n")
        f.write("Vector sum (plagiarism) score: " + str(worst_vector_score))
        f.write("\n ----------------------------------------------------- \n")
        f.write("\n \n Best by vector sum (plagiarism): \n")
        f.write(str(best_text_vector)+ "\n")
        f.write("Vector sum (plagiarism) score: " + str(best_vector_score))
        f.write("\n")


def waiting_voices():
    """
    In order to delay the official output voices when all of the developer 
    information is printed (and to sound cool) this function chooses random 
    voices to speak "loading," "waiting," and "thinking" cues. 
    """
    rand_voice = random.choice(VOICES)
    speech_text = "say -v " + rand_voice + " \" loading \""
    system(speech_text)
    rand_voice = random.choice(VOICES)
    speech_text = "say -v " + rand_voice + " \" waiting \""
    system(speech_text)
    rand_voice = random.choice(VOICES)
    speech_text = "say -v " + rand_voice + " \" thinking \""
    system(speech_text)


nlp = spacy.load("en_core_web_lg")

# Streamlit page config
st.set_page_config(
     page_title="Network Andre",
     page_icon="📹",
 )
st.title("Network Andre")
st.image("images/ericAndreShow.jpg")

developer = st.checkbox('Developer mode (shows all generations and plot)') 
save_output = st.checkbox('Save output in .txt file')
speak_output = st.checkbox('Speak generated output')
episode_title = st.selectbox('Source episode', EPISODES)

andre_doc = Andre(nlp, episode_title)  # Instance of Andre based on episode
andre_doc.set_text()

if st.checkbox("Write your own seed"): 
    seed = st.text_input("Seed")

else:   
    seed = st.selectbox("Seed", SEEDS)

text_len = st.number_input("Approximate length of response (in words, max 500)", \
    min_value=1, max_value=500, value=30)
generations = st.number_input("Number of content generations (max 1000)", \
    min_value=1, max_value=1000, value=50)
temperature = st.slider("Temperature (how much Eric Andre is injected)", \
    min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if st.button('Generate!'):
    
    st.image("images/ericFace.gif")    # Fun loading image 

    generated_text = generate_some_text(seed, text_len) 
    poetry_doc = nlp(generated_text)
    poetry_useful_words = andre_doc.lemmatize_useful_words(poetry_doc)
    poetry_score = sentence_score(poetry_useful_words)

    # worst and best by sentence relevance score 
    worst_text_swapped = ""
    best_text_swapped = ""
    worst_score = 100000000000
    best_score = 0
    worst_speech = ""
    best_speech = ""
    sentence_scores = []

    # worst and best by vector (plagiarism) score
    worst_text_vector = ""
    best_text_vector = ""
    worst_vector_score = 1.0
    best_vector_score = 0.0
    worst_vector_speech = ""
    best_vector_speech = ""
    vector_scores = []

    # Generate new poetry examples via swap for 'generations' number of times 
    #   and keep track of the best and worst for each of the two scoring 
    #   methods to be printed at the end and saved.
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
            # Iteratively prints all outpus for user examination 
            developer_text(generated_text, poetry_score, swapped_output, \
                swapped_score, vector_sum_score)


    # Had these below writing prompts as a helper function, but then it broke.
    st.write("GPT-2 text: \n" + generated_text)
    st.header("The worst and the best:")
    st.subheader("Worst by sentence relevance score:")
    st.write(worst_text_swapped)
    st.write("Internal sentence relevance score: ", worst_score)
    st.subheader("Best by sentence relevance score:")
    st.write(best_text_swapped)
    st.write("Internal sentence relevance score: ", best_score)
    st.subheader("Worst by vector sum (plagiarism):")
    st.write(worst_text_vector)
    st.write("Vector sum (plagiarism) score: ",  worst_vector_score)
    st.subheader("Best by vector sum (plagiarism):")
    st.write(best_text_vector)
    st.write("Vector sum (plagiarism) score: ", best_vector_score)

    if save_output:
        save_poetry(worst_text_swapped, worst_score, best_text_swapped, \
            best_score, worst_text_vector, worst_vector_score, best_text_vector, \
                best_vector_score, generated_text, text_len, generations, \
                    temperature, episode_title)     

    if developer:
        # Plotting internal sentence score vs. vector sum (plagiarism) score 
        fig, score_data = plt.subplots()
        sizes = [10] * len(vector_scores)
        score_data.scatter(sentence_scores, vector_scores, sizes)   
        score_data.set_xlabel("Internal Sentence Relevance Score")
        score_data.set_ylabel("Vector Sum (Plagiarism) Score")
        score_data.set_ybound(0,1.1)
        st.pyplot(fig)

        for i in range(4):
            waiting_voices()


    if speak_output:
        system(worst_speech)
        system(best_speech)
        system(worst_vector_speech) 
        system(best_vector_speech)

    st.image("images/endOfShow.jpg")    # Fun conclusion image 
    
