from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
lemmatizer = WordNetLemmatizer()

def lem(word):
    if word == 'people':
        return 'person'
    return lemmatizer.lemmatize(word, wn.NOUN)