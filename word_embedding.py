"""

https://levelup.gitconnected.com/introduction-to-natural-language-processing-nlp-in-pytorch-8b7344c9dfec for h5py understanding 
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

index = {word: i for i, word in enumerate(english_words)}


def similarity_score(word_1, word_2):
    """
    This function was taken from the repo at top of file
    """

    if (word_1 not in index) | (word_2 not in index):
        return 0 
    score = np.dot(normalized_embeddings[index[word_1], :], \
        normalized_embeddings[index[word_2], :])
    return score

def sentence_score(sentence):
    """
    Pass in a sentence list where the words are only meaningful words 
    wrote this myself 
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
    wrote this myself 
    sentence inputs are lists of meaningful words
    """
    vector_1 = np.array([0]*300)   # these word embeddings have 300 dimensions 
    vector_2 = np.array([0]*300)

    for word in sentence_1: 
        if word not in index:
            vector_1 = vector_1 + np.array([0]*300)
        else:
            vector_1 = vector_1 + np.array(normalized_embeddings[index[word], :])

    for word in sentence_2:
        if word not in index:
            vector_2 = vector_2 + np.array([0]*300)
        else:
            vector_2 = vector_2 + np.array(normalized_embeddings[index[word], :])


    normalized_vector_1 = vector_1 / np.linalg.norm(vector_1)    
    normalized_vector_2 = vector_2 / np.linalg.norm(vector_2)

    return np.dot(normalized_vector_1, normalized_vector_2)







