import spacy
from spacy.matcher import Matcher


# nlp = spacy.load("en_core_web_sm")
# matcher = Matcher(nlp.vocab)



class Andre:
    def __init__(self, nlp, text):
        self.nlp = nlp 
        self.text = text
        self.processed = nlp(text) 

    def sentences(self):
        config = {"punct_chars": ['\n']}
        # custom_nlp.add_pipe(set_custom_boundaries, before='parser')
        self.nlp.add_pipe("sentencizer", config=config)
        for sent in self.processed.sents:
            print('next sentence:')
            print(sent) 

# def set_custom_boundaries(doc):
#      # Adds support to use `\n` as the delimiter for sentence detection
#     for token in doc[:-1]:
#         if token.text == '\n':
#             doc[token.i+1].is_sent_start = True
#     return doc


# custom_new_line_doc = custom_nlp(text)
# custom_new_line_sentences = list(custom_new_line_doc.sents)
# for sentence in custom_new_line_sentences:
#     print(sentence)



'''


# Process the text

doc = nlp(text)


for token in doc:
    # Get the token text, part-of-speech tag and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_    
    # This is for formatting only
    print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")
    print(spacy.explain(token_pos))


# Iterate over the predicted entities
print("\nentity stuff")
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)
    print(spacy.explain(ent.label_))



# Get the span for "iPhone X"
print("\n")
ladies_and_gentlemen = doc[0:3]

# Print the span text
print("Missing entity:", ladies_and_gentlemen.text)
'''