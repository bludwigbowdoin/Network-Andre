import os
import spacy
from spacy.matcher import Matcher
import re 
import random
from word_embedding import *

SPECIAL_POS = ["ADJ", "ADV", "INTJ", "NOUN", "VERB"]


class Andre:
    def __init__(self, nlp, title):
        self.nlp = nlp 
        self.doc = None
        self.title = title
        self.full_text = ''

    def set_text(self):
        # Data prep
        with open(self.title) as f:
            contents = re.sub(r'someword=|\,.*|\#.*','', f.read())
            contents = re.sub(r'\n+', '\n', contents).strip()

        self.full_text = contents
        self.doc = self.nlp(self.full_text)

    def sentences(self):
        config = {"punct_chars": ['\n']}
        self.nlp.add_pipe("sentencizer", config=config)
        return self.doc.sents

    def no_stop_words(self):
        token_list = []
        for token in self.doc:
            if not token.is_stop:
                token_list.append(token)
        
        return token_list

    def tokens_pos(self):
        # for all tokens in self, organize them by part-of-speech
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
        output_list = []

        for token in poetry.doc: 
            if token.pos_ in SPECIAL_POS: 
                output_list.append(token.lemma_)

        return output_list 


    def swap_within_pos(self, poetry, temperature):
        output_string = ""
        # initialize new poem output string 
        # go through tokens of self 
        # if token.pos_ = ADJ, ADV, INTJ, NOUN, VERB, 
        #   then randomly select one of those POS from source_andre
        #   add it to output string 
        # else 
        #   add what was previously there to output string 
        other_pos_dict = self.tokens_pos()
        for token in poetry.doc: 
            if token.pos_ in SPECIAL_POS and random.random() < temperature: 
                token_to_add = random.choice(other_pos_dict[token.pos_])
                output_string = output_string + " " + token_to_add
            else:
                output_string = output_string + " " + token.text 
        return output_string 



# nlp = spacy.load("en_core_web_sm")
# matcher = Matcher(nlp.vocab)

# poetry_doc = nlp("the rose is the most beautiful color of the shirt that he kicks down the road and on and on forever lovely.")

# andre = Andre(nlp, "season25ep1356.txt")
# andre.set_text()

# output = andre.swap_within_pos(poetry_doc, 0.3)

# print(andre.lemmatize_useful_words(poetry_doc))

# print(sentence_score(andre.lemmatize_useful_words(poetry_doc)))

# 





