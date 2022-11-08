import spacy
from spacy.matcher import Matcher

from andre import Andre 



def main():
    """

    """
    with open('season2ep1.txt') as f:
        contents = f.read()
    with open('season2ep3.txt') as f:
        contents = f.read()
    with open('season5ep5.txt') as f:
        contents = f.read()
    with open('season5ep6.txt') as f:
        contents = f.read()


    text = contents
    custom_nlp = spacy.load('en_core_web_sm')
    andre_example = Andre(custom_nlp, text)
    # andre_example.print_sentences()

    andre_example.speak_text()
    # print(andre_example.no_stop_words())
    


if __name__ == '__main__':
    main()
