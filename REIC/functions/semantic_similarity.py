import re
from functions.semantic_similarity_fasttext import get_similarity as fasttext_similarity

# WordNet
from nltk.corpus import wordnet as wn


def get_similarity(word1, word2, threshold):
    similarity = fasttext_similarity(word1, word2)
    if similarity < threshold and similarity > 0.5:
        if word1 != 'object' or word2 != 'object':
            if is_hyper(word1, word2):
                return threshold
    return similarity


def is_hyper(word1, word2):
    def is_hyper_single(w1, w2):
        # if there is hyper-relation between w1 and w2
        word1_set = wn.synsets(w1, pos=wn.NOUN)[:2]
        word2_set = wn.synsets(w2, pos=wn.NOUN)[:2]
        for n1 in word1_set:
            for n2 in word2_set:
                n = n1.lowest_common_hypernyms(n2)
                if n[0] == n1 or n[0] == n2:
                    return True
        return False
    word1 = re.sub(r'[^a-z]', ' ', word1)
    word2 = re.sub(r'[^a-z]', ' ', word2)
    if not (word1.__contains__(' ') or word2.__contains__(' ')):
        return is_hyper_single(word1, word2)
    word1_list = [word1] if len(wn.synsets(word1, pos=wn.NOUN)) != 0 else word1.split(' ')
    word2_list = [word2] if len(wn.synsets(word2, pos=wn.NOUN)) != 0 else word2.split(' ')
    return max([is_hyper_single(w1, w2) for w1 in word1_list for w2 in word2_list])

