import os
current_dir = os.path.dirname(os.getcwd())

from gensim.models import KeyedVectors
import re
model = KeyedVectors.load_word2vec_format(f'{current_dir}/functions/wiki-news-300d-1M.vec')


def split_word(word):
    # split words or return word
    word = word.lower()
    word = re.sub(r'[^a-z]', ' ', word)
    if word.__contains__(' '):
        return word.split(' ')
    else:
        return [word]


def get_similarity_single(w1, w2):
    # get similarity from single words
    if w1 in model.key_to_index and w2 in model.key_to_index:
        return model.similarity(w1, w2)
    else:
        return 1.0 if w1 == w2 else 0.0


def get_similarity(word1, word2):
    # get similarity
    word1_list = split_word(word1)
    word2_list = split_word(word2)
    return max([get_similarity_single(w1, w2) for w1 in word1_list for w2 in word2_list])
