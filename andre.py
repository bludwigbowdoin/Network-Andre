import spacy
from spacy.matcher import Matcher


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


with open('season2ep1.txt') as f:
    contents = f.read()

text = contents

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