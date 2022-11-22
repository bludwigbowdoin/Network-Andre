"""
Bjorn Ludwig
CSCI 3725
M6: Poetry Slam
11/22/2022

This file contains the Andre class, which defines a number of useful functions
dependent on tools from spaCy as well as word_embedding. This class also 
enables functionality to work between two spaCy-processed docs. 
"""

import os
import re 
import random
from word_embedding import *

# Meaningful parts-of-speech
SPECIAL_POS = ["ADJ", "ADV", "INTJ", "NOUN", "VERB"]

class Andre:
    def __init__(self, nlp, title):
        """
        Andre class which contains the spaCy functionality of the system. 

        Args: 
            nlp (spaCy nlp): the natural language processor.
            title (str): title of the episode to build instance upon.
        """
        self.nlp = nlp 
        self.doc = None
        self.title = title
        self.full_text = ''


    def set_text(self):
        """
        Processes the episode's .txt file, sets the instance's text, and sets 
        the nlp doc for the instance.

        Args:
            none 
        """
        with open("episodes/" + self.title) as f:
            contents = re.sub(r'someword=|\,.*|\#.*','', f.read())
            # clean up extraneous newlines 
            contents = re.sub(r'\n+', '\n', contents).strip() 

        self.full_text = contents
        self.doc = self.nlp(self.full_text)


    def sentences(self):
        """
        Incorporates newlines as sentence ends. 

        Args:
            none 
        """
        config = {"punct_chars": ['\n']}
        self.nlp.add_pipe("sentencizer", config=config)
        return self.doc.sents


    def tokens_pos(self):
        """
        Organize all the tokens in the instance by part of speech. Note:
        duplicates are intentionally allowed so that randomly selecting from 
        these lists maintains the original distribution of the words in the 
        source episode. 

        Args:
            none 
        """
        pos_dict = {}
        for token in self.doc:
            if token.pos_ not in pos_dict.keys():
                word_list = []
                word_list.append(token.text)
                pos_dict[token.pos_] = word_list
            else: 
                word_list = pos_dict[token.pos_]
                word_list.append(token.text)
                pos_dict[token.pos_] = word_list
        return pos_dict


    def lemmatize_useful_words(self, poetry):
        """
        Lemmatize (shorten to base form) each token in the given poetry 
        doc. 

        Args:
            poetry (spaCy doc): a relevant spaCy doc to be examined alongside 
                                    this instance of Andre.  
        """
        output_list = []

        for token in poetry.doc: 
            if token.pos_ in SPECIAL_POS: 
                output_list.append(token.lemma_)

        return output_list 


    def swap_within_pos(self, poetry, temperature):
        """
        Iterate through tokens of the poetry doc, inject random interjections 
        in the style of Eric Andre, and for particular parts-of-speech, 
        swap in a word of the same part-of-speech from the source episode. 

        Args:
            poetry (spaCy doc): a relevant spaCy doc to be examined alongside 
            temperature (float): the probability 0.0 to 1.0 that a given 
                                    special word gets swapped
        """
        output_string = ""
        other_pos_dict = self.tokens_pos()

        for token in poetry.doc: 
            if random.random() < (temperature / 10):   # random interjection
                token_to_add = random.choice(other_pos_dict["INTJ"])
                output_string = output_string + " " + token_to_add

            elif token.pos_ in SPECIAL_POS and random.random() < temperature: 
                token_to_add = random.choice(other_pos_dict[token.pos_])
                output_string = output_string + " " + token_to_add

            else:
                output_string = output_string + " " + token.text 

        return output_string 

