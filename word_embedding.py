"""
Bjorn Ludwig
CSCI 3725
M6: Poetry Slam
11/22/2022

This file contains 
for h5py understanding 

levelup.gitconnected.com/introduction-to-natural-language-processing-nlp-in-pytorch-8b7344c9dfec 
"""

import h5py
import numpy as np

with h5py.File('mini.h5', 'r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]

# Restrict our vocabulary to just the English words
english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
english_embeddings = all_embeddings[english_word_indices]

# Normalize the embedding vectors, emphasizing direction over magnitude
norms = np.linalg.norm(english_embeddings, axis=1)
normalized_embeddings = english_embeddings.astype('float32') / \
    norms.astype('float32').reshape([-1, 1])

# Dictionary mapping the word embedding vector entries to corresponding word 
index = {word: i for i, word in enumerate(english_words)}


def similarity_score(word_1, word_2):
    """
    This function was taken from the repo at top of file. It simply returns 
    the simialrity of two input words via the dot product of their embeddings. 
    I added the conditional in case some of the words used do not exist in the 
    word embedding data set. 

    Args: 
        word_1 (str): first word to compare to the second word
        word_2 (str): second word to compare to the first word
    """

    if (word_1 not in index) | (word_2 not in index):
        return 0 
    score = np.dot(normalized_embeddings[index[word_1], :], \
        normalized_embeddings[index[word_2], :])
    return score

def sentence_score(sentence):
    """
    Sums all similarity scores of distinct pairs of words in the sentence 
    input (wrote this myself). 

    Args: 
        sentence (arr): an array of "important" words from a given sentence 
    """
    score = 0
    for first_word_index in range(len(sentence)): 
        first_word = sentence[first_word_index]
        for second_word_index in range(1+first_word_index, len(sentence)):
            second_word = sentence[second_word_index]
            pair_score = similarity_score(first_word, second_word)
            score += pair_score

    return score 

def vector_addition_score(sentence_1, sentence_2):
    """
    Adds all individual word vectors (tip to tail) from one sentence, 
    normalizes that sum, then adds all individual word vectors from the other 
    sentence, normalizes that sum, then computes dot product similarity 
    between the two sums. This is a good measure 0 to 1 of plagiarism (wrote 
    this myself).

    Args: 
        sentence_1 (arr): an array of "important" words from given sentence 1
        sentence_2 (arr): an array of "important" words from given sentence 2
    """
    vector_1 = np.array([0]*300)   # these word embeddings have 300 dimensions 
    vector_2 = np.array([0]*300)

    for word in sentence_1: 
        if word not in index:   # treat words not found as the zero vector
            vector_1 = vector_1 + np.array([0]*300)
        else:
            vector_1 = vector_1 + np.array(normalized_embeddings[index[word], :])

    for word in sentence_2:
        if word not in index:   # treat words not found as the zero vector
            vector_2 = vector_2 + np.array([0]*300)
        else:
            vector_2 = vector_2 + np.array(normalized_embeddings[index[word], :])


    normalized_vector_1 = vector_1 / np.linalg.norm(vector_1)    
    normalized_vector_2 = vector_2 / np.linalg.norm(vector_2)

    return np.dot(normalized_vector_1, normalized_vector_2)

